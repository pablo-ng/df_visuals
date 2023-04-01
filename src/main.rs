use anyhow::{anyhow, Result};
use bevy::{
    prelude::*,
    sprite::{Anchor, MaterialMesh2dBundle},
    text::BreakLineOn,
    window::{WindowMode, WindowResolution},
};
use clap::Parser;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, SampleRate, Stream, StreamConfig,
};
use mel_filter::{mel, NormalizationFactor};
use rand::Rng;
use rubato::{FftFixedIn, Resampler};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::{
    collections::VecDeque,
    f32::consts::TAU,
    ops::Range,
    sync::{Arc, RwLock},
};

#[derive(Parser, Resource)]
struct Cli {
    input_device_index: usize,
    artist_name: String,

    #[arg(
        short = 's',
        default_value_t = 22050,
        help = "Sample rate used for audio processing"
    )]
    sample_rate: u32,

    #[arg(
        short = 'w',
        default_value_t = 2048,
        help = "Window size used for audio processing"
    )]
    window_size: usize,

    #[arg(short, long, default_value_t = 35)]
    fps: u32,

    #[arg(
        long,
        default_value_t = 22050,
        help = "Input device sample rate. Will be resamples to sample_rate"
    )]
    device_sample_rate: u32,

    #[arg(long, default_value_t = 512, help = "Input device buffer size")]
    device_buffer_size: u32,

    #[arg(
        long,
        default_value_t = 1,
        help = "Input device number of channels. Will be remapped to 1 channel."
    )]
    device_channels: u16,

    #[arg(long, default_value_t = 150.)]
    particle_amp_threshold: f32,
    #[arg(long, default_value_t = 0.2)]
    particle_logo_probability: f32,
}

const BACKGROUND_COLOR: Color = Color::rgb(
    5. / u8::MAX as f32,
    53. / u8::MAX as f32,
    23. / u8::MAX as f32,
);
const PARTICLE_NORMAL_COLOR_RANGES: [Range<f32>; 3] = [0.0..0.4, 0.4..1.0, 0.0..0.4];
const PARTICLE_NORMAL_RADIUS: f32 = 10.0;
const CIRCLE_COLOR: Color = Color::rgb(
    3. / u8::MAX as f32,
    37. / u8::MAX as f32,
    23. / u8::MAX as f32,
);
const CIRCLE_RADIUS: f32 = 200.;
const FONT_SIZE: f32 = 200.;

#[derive(Component)]
struct Particle {
    velocity: Vec2,
}

#[derive(Component)]
struct ArtistNameText;

#[derive(Component)]
struct CenterCircle;

#[derive(Resource)]
struct LogoImage {
    handle: Handle<Image>,
}

#[derive(Resource, Debug)]
struct Dynamics {
    low_freq_amp: f32,
    scaling: f32,
    movement: f32,
    circle_radius: f32,
}

#[derive(Resource)]
struct DynamicsResource {
    mel_basis: Vec<Vec<f32>>,
    fft_runner: Arc<dyn Fft<f32>>,
}

static AUDIO_BUFFER: RwLock<VecDeque<f32>> = RwLock::new(VecDeque::new());

fn update_dynamics(
    mut dynamics: ResMut<Dynamics>,
    args: Res<Cli>,
    dynamics_resource: Res<DynamicsResource>,
) {
    let mut buffer = {
        // lock AUDIO_BUFFER as short as possible
        let r = AUDIO_BUFFER.read().unwrap();
        let buffer = r
            .iter()
            .map(|value| Complex::new(*value, 0.0))
            .collect::<Vec<_>>();
        buffer
    };

    assert!(buffer.len() == args.window_size); // TODO remove?

    // compute the fft
    // TODO use planner with power and abs directly
    dynamics_resource.fft_runner.process(&mut buffer);

    // compute the power spectrum
    let power: Vec<_> = buffer.into_iter().map(|val| val.norm().powi(2)).collect();

    // apply the mel filterbank to the power spectrum
    let mel_power = dynamics_resource.mel_basis.iter().map(|values| {
        values
            .iter()
            .zip(power.iter())
            .map(|(x, y)| x * y)
            .sum::<f32>()
    });

    // compute the log mel spectrogram
    let log_mel_power = mel_power.map(|val| 10. * val.log10());

    // compute the dynamics
    let imin = -500.; // TODO choose values wisely
    let imax = 500.;
    let omin = 0.6;
    let omax = 1.4;
    let ratio = (omax - omin) / (imax - imin);
    dynamics.low_freq_amp = log_mel_power.take(16).sum::<f32>().clamp(imin, imax);
    dynamics.scaling = dynamics.low_freq_amp * ratio - imin * ratio + omin;
    dynamics.movement = dynamics.low_freq_amp / 100.;
    dynamics.circle_radius = CIRCLE_RADIUS * dynamics.scaling;
}

fn create_particles(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    dynamics: Res<Dynamics>,
    logo_image: Res<LogoImage>,
    args: Res<Cli>,
) {
    // add new particles
    if dynamics.low_freq_amp > args.particle_amp_threshold {
        let mut rng = rand::thread_rng();
        let theta: f32 = rng.gen_range(0.0..TAU);
        let velocity = Vec2::new(
            dynamics.circle_radius * theta.cos(),
            dynamics.circle_radius * theta.sin(),
        );
        let position = Vec3::from((velocity, 0.0));

        if rng.gen_range(0.0..1.0) < args.particle_logo_probability {
            commands.spawn((
                SpriteBundle {
                    texture: logo_image.handle.clone(),
                    transform: Transform::from_translation(position),
                    ..default()
                },
                Particle { velocity },
            ));
        } else {
            let r = rng.gen_range(PARTICLE_NORMAL_COLOR_RANGES[0].clone());
            let g = rng.gen_range(PARTICLE_NORMAL_COLOR_RANGES[1].clone());
            let b = rng.gen_range(PARTICLE_NORMAL_COLOR_RANGES[2].clone());
            commands.spawn((
                MaterialMesh2dBundle {
                    mesh: meshes
                        .add(shape::Circle::new(PARTICLE_NORMAL_RADIUS).into())
                        .into(),
                    material: materials.add(ColorMaterial::from(Color::rgb(r, g, b))),
                    transform: Transform::from_translation(position),
                    ..default()
                },
                Particle { velocity },
            ));
        }
    }
}

fn update_particles(
    mut commands: Commands,
    mut query: Query<(Entity, &Particle, &mut Transform)>,
    window_query: Query<&Window>,
    dynamics: Res<Dynamics>,
    time: Res<FixedTime>,
) {
    let window = window_query.get_single().unwrap();
    let max_x = window.width() / 2.;
    let max_y = window.height() / 2.;
    for (entity, particle, mut transform) in &mut query {
        // move particles
        let time_step = time.period.as_secs_f32();
        transform.translation.x += particle.velocity.x * time_step * dynamics.movement;
        transform.translation.y += particle.velocity.y * time_step * dynamics.movement;

        // clip particle position to circle radius
        if transform.translation.length() < dynamics.circle_radius {
            transform.translation = dynamics.circle_radius * transform.translation.normalize();
        }

        // remove particles that are off the screen
        if transform.translation.x.abs() > max_x || transform.translation.y.abs() > max_y {
            commands.entity(entity).despawn();
        }
    }
}

fn update_circle_and_text(
    mut query: Query<&mut Transform, Or<(With<CenterCircle>, With<ArtistNameText>)>>,
    dynamics: Res<Dynamics>,
) {
    for mut transform in &mut query {
        transform.scale = Vec3::new(dynamics.scaling, dynamics.scaling, 0.);
    }
}

fn init_audio_input(
    device: &Device,
    sample_rate: u32,
    window_size: usize,
    device_channels: u16,
    device_sample_rate: u32,
    device_buffer_size: u32,
) -> Result<Stream> {
    let supported_input_configs = device.supported_input_configs()?.collect::<Vec<_>>();
    let config = StreamConfig {
        channels: device_channels,
        sample_rate: SampleRate(device_sample_rate),
        buffer_size: cpal::BufferSize::Fixed(device_buffer_size),
    };

    let device_channels: usize = device_channels.into();

    {
        // fill AUDIO_BUFFER with 0
        let mut w = AUDIO_BUFFER.write().unwrap();
        w.extend(vec![0.; window_size]);
        assert!(w.len() == window_size);
    }

    // TODO still not optimal
    let mut resampler = FftFixedIn::<f32>::new(
        device_sample_rate as usize,
        sample_rate as usize,
        480, // TODO always? -> no!!
        1,
        1,
    )
    .unwrap();

    device
        .build_input_stream(
            &config,
            move |data: &[i16], _: &cpal::InputCallbackInfo| {
                // remap to one channel
                // no need to use chunks_exact as exact length is asserted above
                // TODO what if one channel is inverted?
                let new_samples = data
                    .chunks_exact(device_channels)
                    .map(|values| {
                        values.iter().map(|value| f32::from(*value)).sum::<f32>()
                            / (device_channels as f32)
                    })
                    .collect::<Vec<f32>>();

                // assert!(new_samples == 480); // TODO!

                // resample to sample_rate
                // TODO This is a convenience wrapper for process_into_buffer that allocates the output buffer with each call. For realtime applications, use process_into_buffer with a buffer allocated by output_buffer_allocate instead of this function.
                let new_samples = resampler.process(&[new_samples], None).unwrap();
                let new_samples = &new_samples[0];

                // assert!(
                //     new_samples.len() == HOP_LENGTH,
                //     "got {} new_samples but expected {}",
                //     new_samples.len(),
                //     HOP_LENGTH,
                // );

                {
                    // lock AUDIO_BUFFER as short as possible
                    let mut w = AUDIO_BUFFER.write().unwrap();
                    w.drain(..new_samples.len());
                    w.extend(new_samples);
                }
            },
            |err| {
                panic!("{}", err);
            },
            None,
        )
        .map_err(|err| match err {
            cpal::BuildStreamError::StreamConfigNotSupported => anyhow!(
                "Unsupported configuration for this device.\n\
                Requested configuration: {:?};\n\
                Supported configuration: {:?}",
                config,
                supported_input_configs
            ),
            _ => anyhow!(err),
        })
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    asset_server: Res<AssetServer>,
    args: Res<Cli>,
) {
    commands.spawn(Camera2dBundle::default());

    // add center circle
    commands.spawn((
        MaterialMesh2dBundle {
            mesh: meshes.add(shape::Circle::new(CIRCLE_RADIUS).into()).into(),
            material: materials.add(ColorMaterial::from(CIRCLE_COLOR)),
            transform: Transform::from_translation(Vec3::new(0.0, 0.0, 0.0)),
            ..default()
        },
        CenterCircle,
    ));

    // add artist name text
    commands.spawn((
        Text2dBundle {
            text: Text {
                sections: vec![TextSection::new(
                    args.artist_name.clone(),
                    TextStyle {
                        font: asset_server.load("FiraSans-Bold.ttf"),
                        font_size: FONT_SIZE, // TODO drop shadow?
                        color: Color::WHITE,
                        ..default()
                    },
                )],
                alignment: TextAlignment::Center,
                linebreak_behaviour: BreakLineOn::WordBoundary,
            },
            text_anchor: Anchor::Center,
            ..default()
        },
        ArtistNameText,
    ));

    // load logo image
    commands.insert_resource(LogoImage {
        handle: asset_server.load("df_logo.png"),
    });
}

fn main() -> Result<()> {
    // get and list input devices to the user
    println!("Available Input Devices:");
    let host = cpal::default_host();
    let devices = host.input_devices()?.collect::<Vec<_>>();
    for (i, device) in devices.iter().enumerate() {
        println!(
            "{}: {}",
            i,
            device.name().unwrap_or(String::from("unknown"))
        );
    }

    // parse args
    let args = Cli::parse();
    let time_step = 1.0 / args.fps as f32;

    // start audio input stream
    let device = devices
        .get(args.input_device_index)
        .ok_or(anyhow!("Invalid input device index"))?;
    let stream = init_audio_input(
        &device,
        args.sample_rate,
        args.window_size,
        args.device_channels,
        args.device_sample_rate,
        args.device_buffer_size,
    )?;
    stream.play()?;

    // compute the mel filterbank
    let n_fft = (args.window_size - 1) * 2; // TODO correct?
    let mel_basis = mel::<f32>(
        args.sample_rate as usize,
        n_fft,
        Some(128), // TODO good?
        None,
        None,
        false,
        NormalizationFactor::One,
    );

    // init FftRunner
    let mut planner = FftPlanner::new();
    let fft_runner = planner.plan_fft_forward(args.window_size);

    App::new()
        // TODO remove default plugins, use only required plugins
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: String::from("DF Visuals"),
                // mode: WindowMode::Fullscreen, // TODO uncomment
                resizable: false,
                resolution: WindowResolution::new(1200., 800.), // TODO comment
                // window_level: WindowLevel::AlwaysOnTop, // TODO
                ..default()
            }),
            ..default()
        }))
        .insert_resource(args)
        .insert_resource(ClearColor(BACKGROUND_COLOR))
        .insert_resource(Dynamics {
            low_freq_amp: 0.,
            scaling: 0.,
            movement: 0.,
            circle_radius: 0.,
        })
        .insert_resource(DynamicsResource {
            mel_basis,
            fft_runner,
        })
        .add_startup_system(setup)
        .add_systems(
            (
                update_dynamics,
                update_circle_and_text.after(update_dynamics),
                create_particles.after(update_dynamics),
                update_particles.after(create_particles),
            )
                .in_schedule(CoreSchedule::FixedUpdate),
        )
        .insert_resource(FixedTime::new_from_secs(time_step))
        .add_system(bevy::window::close_on_esc)
        .run();

    Ok(())
}
