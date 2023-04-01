use anyhow::{anyhow, Result};
use bevy::{
    prelude::*,
    sprite::{Anchor, MaterialMesh2dBundle},
    text::{BreakLineOn, Text2dBounds},
    window::{WindowLevel, WindowMode, WindowResolution},
};
use clap::Parser;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, FromSample, SampleFormat, SampleRate, SizedSample, Stream, StreamConfig,
};
use mel_filter::{mel, NormalizationFactor};
use rand::Rng;
use rubato::{FftFixedOut, Resampler};
use rustfft::{num_complex::Complex, Fft, FftPlanner};
use std::{
    collections::VecDeque,
    f32::consts::TAU,
    ops::Range,
    sync::{Arc, RwLock},
};

const WINDOW_TITLE: &str = "DF Visuals";
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

// TODO choose values wisely
const AMP_RANGE: [f32; 2] = [-700., 500.];
const SCALING_RANGE: [f32; 2] = [0.6, 1.4];
const MOVEMENT_SPEED: f32 = 1. / 60.;
const SMOOTHING_FACTOR: f32 = 0.6;
const AMP_SCALING_RATIO: f32 =
    (SCALING_RANGE[1] - SCALING_RANGE[0]) / (AMP_RANGE[1] - AMP_RANGE[0]);

#[derive(Parser, Resource)]
struct Cli {
    input_device_index: usize,
    artist_name: String,
    #[arg(short = 's', default_value_t = 22050)]
    sample_rate: u32,
    #[arg(short = 'w', default_value_t = 2048)]
    window_size: usize,
    #[arg(short, long, default_value_t = 35)]
    fps: u32,
    #[arg(long, action)]
    windowed: bool,
    #[arg(long, default_value_t = 0)]
    device_input_config: usize,
    #[arg(long, default_value_t = 22050)]
    device_sample_rate: u32,
    #[arg(long, default_value_t = 512)]
    device_buffer_size: u32,
    #[arg(long, default_value_t = 150.)]
    particle_amp_threshold: f32,
    #[arg(long, default_value_t = 0.2)]
    particle_logo_probability: f32,
    #[arg(long, default_value_t = 200.)]
    circle_radius: f32,
    #[arg(long, default_value_t = 200.)]
    font_size: f32,
}

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

    assert!(buffer.len() == args.window_size); // TODO should not be needed

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
    dynamics.low_freq_amp = log_mel_power
        .take(16)
        .sum::<f32>()
        .clamp(AMP_RANGE[0], AMP_RANGE[1]);
    dynamics.scaling = (1. - SMOOTHING_FACTOR) * dynamics.scaling
        + SMOOTHING_FACTOR
            * (dynamics.low_freq_amp * AMP_SCALING_RATIO - AMP_RANGE[0] * AMP_SCALING_RATIO
                + SCALING_RANGE[0]);
    dynamics.movement = MOVEMENT_SPEED * dynamics.low_freq_amp;
    dynamics.circle_radius = args.circle_radius * dynamics.scaling;
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

fn init_audio_input<T: SizedSample>(
    device: &Device,
    sample_rate: u32,
    window_size: usize,
    config: StreamConfig,
) -> Result<Stream>
where
    f32: FromSample<T>,
{
    {
        // fill AUDIO_BUFFER with 0
        let mut w = AUDIO_BUFFER.write().unwrap();
        w.extend(vec![0.; window_size]);
        assert!(w.len() == window_size);
    }

    let mut resampler = if config.sample_rate.0 != sample_rate {
        println!(
            "audio will be resampled from {} to {}",
            config.sample_rate.0, sample_rate
        );
        Some(
            FftFixedOut::<f32>::new(
                config.sample_rate.0 as usize,
                sample_rate as usize,
                480, // TODO is not always the same
                1,
                1,
            )
            .unwrap(),
        )
    } else {
        None
    };

    let stream = device.build_input_stream(
        &config,
        move |data: &[T], _: &cpal::InputCallbackInfo| {
            // remap to one channel
            // no need to use chunks_exact as exact length is asserted above
            // TODO what if one channel is inverted?
            let new_samples = data
                .chunks_exact(config.channels as usize)
                .map(|values| {
                    values
                        .iter()
                        .map(|value| f32::from_sample_(*value))
                        .sum::<f32>()
                        / (config.channels as f32)
                })
                .collect::<Vec<f32>>();

            // resample to sample_rate
            // TODO This is a convenience wrapper for process_into_buffer that allocates the output buffer with each call. For realtime applications, use process_into_buffer with a buffer allocated by output_buffer_allocate instead of this function.
            let new_samples = if let Some(ref mut resampler) = &mut resampler {
                let mut new_samples = resampler.process(&[new_samples], None).unwrap();
                new_samples.pop().unwrap()
            } else {
                new_samples
            };

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
    )?;

    Ok(stream)
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
            mesh: meshes
                .add(shape::Circle::new(args.circle_radius).into())
                .into(),
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
                        font_size: args.font_size, // TODO drop shadow?
                        color: Color::WHITE,
                        ..default()
                    },
                )],
                alignment: TextAlignment::Center,
                linebreak_behaviour: BreakLineOn::WordBoundary,
            },
            text_anchor: Anchor::Center,
            text_2d_bounds: Text2dBounds {
                size: Vec2::new(args.circle_radius * 2., args.circle_radius * 2.),
            },
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
    println!(
        "Supported Input Configs for device {}:",
        args.input_device_index
    );
    let supported_input_configs = device.supported_input_configs()?.collect::<Vec<_>>();
    for (i, input_config) in supported_input_configs.iter().enumerate() {
        println!("{}: {:?}", i, input_config);
    }
    let input_config = supported_input_configs
        .get(args.device_input_config)
        .ok_or(anyhow!("Invalid input config selected"))?
        .clone()
        .with_sample_rate(SampleRate(args.device_sample_rate));
    let config = StreamConfig {
        channels: input_config.channels(),
        sample_rate: input_config.sample_rate(),
        buffer_size: cpal::BufferSize::Fixed(args.device_buffer_size),
    };
    let stream = match input_config.sample_format() {
        // TODO use macro
        SampleFormat::I8 => {
            init_audio_input::<i8>(&device, args.sample_rate, args.window_size, config)?
        }
        SampleFormat::I16 => {
            init_audio_input::<i16>(&device, args.sample_rate, args.window_size, config)?
        }
        SampleFormat::I32 => {
            init_audio_input::<i32>(&device, args.sample_rate, args.window_size, config)?
        }
        SampleFormat::I64 => {
            init_audio_input::<i64>(&device, args.sample_rate, args.window_size, config)?
        }
        SampleFormat::U8 => {
            init_audio_input::<u8>(&device, args.sample_rate, args.window_size, config)?
        }
        SampleFormat::U16 => {
            init_audio_input::<u16>(&device, args.sample_rate, args.window_size, config)?
        }
        SampleFormat::U32 => {
            init_audio_input::<u32>(&device, args.sample_rate, args.window_size, config)?
        }
        SampleFormat::U64 => {
            init_audio_input::<u64>(&device, args.sample_rate, args.window_size, config)?
        }
        SampleFormat::F32 => {
            init_audio_input::<f32>(&device, args.sample_rate, args.window_size, config)?
        }
        SampleFormat::F64 => {
            init_audio_input::<f64>(&device, args.sample_rate, args.window_size, config)?
        }
        _ => todo!(),
    };
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
            primary_window: Some(match args.windowed {
                false => Window {
                    title: String::from(WINDOW_TITLE),
                    mode: WindowMode::Fullscreen,
                    resizable: false,
                    window_level: WindowLevel::AlwaysOnTop,
                    ..default()
                },
                true => Window {
                    title: String::from(WINDOW_TITLE),
                    resizable: true,
                    resolution: WindowResolution::new(1200., 800.),
                    ..default()
                },
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
