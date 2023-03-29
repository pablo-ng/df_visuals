use anyhow::{anyhow, Result};
use bevy::{
    prelude::*,
    sprite::{Anchor, MaterialMesh2dBundle},
    text::BreakLineOn,
    window::{WindowLevel, WindowResolution},
};
use clap::Parser;
use cpal::{
    traits::{DeviceTrait, HostTrait, StreamTrait},
    Device, SampleRate, Stream, StreamConfig,
};
use rand::Rng;
use rustfft::{num_complex::Complex, FftPlanner};
use std::{collections::VecDeque, f32::consts::TAU, sync::RwLock};

#[derive(Parser)]
struct Cli {
    input_device_index: usize,
    artist_name: String,
    #[arg(short, default_value_t = 22050)]
    sample_rate: u32,
    #[arg(short, default_value_t = 256)]
    buffer_size: u32,
    #[arg(short, default_value_t = 1)]
    channels: u16,
    #[arg(long, default_value_t = 35)]
    fps: u32, // TODO test that this matches by printing time delta, TODO higher fps = more particles right now
    #[arg(long, default_value_t = 0.8)]
    particle_amp_threshold: f32,
    #[arg(long, default_value_t = 0.2)]
    particle_logo_probability: f32,
    #[arg(long, default_value_t = String::from("#053517"))]
    background_color: String,
    #[arg(long, default_value_t = String::from("#FF8080"))]
    particle_normal_color: String, // TODO is overridden by random values
    #[arg(long, default_value_t = 10.0)]
    particle_normal_radius: f32,
    #[arg(long, default_value_t = String::from("#032517"))]
    circle_color: String,
    #[arg(long, default_value_t = 100.0)]
    circle_radius: f32,
}

#[derive(Resource)]
struct Params {
    time_step: f32,
    artist_name: String,
    particle_amp_threshold: f32,
    particle_logo_probability: f32,
    particle_normal_color: Color,
    particle_normal_radius: f32,
    circle_color: Color,
    circle_radius: f32,
    sample_rate: u32,
}

const N_SAMPLES: usize = 480; // TODO always 480?

// TODO WARN bevy_text::glyph_brush: warning[B0005]: Number of font atlases has exceeded the maximum of 16. Performance and memory usage may suffer.

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

#[derive(Resource)]
struct LowFreqAmp(f32);

static AUDIO_BUFFER: RwLock<VecDeque<f32>> = RwLock::new(VecDeque::new());

const WINDOW_SIZE: usize = 4096;
const HOP_LENGTH: usize = 512;
const AUDIO_BUFFER_MAX_SIZE: usize = WINDOW_SIZE * 16;

fn get_low_freq_amp(mut low_freq_amp: ResMut<LowFreqAmp>, params: Res<Params>) {
    // TODO use reader and not drain but use circular?

    // TODO compute the update rate, could also add a flag "newdata" to the buffer

    let mut buffer = {
        // lock AUDIO_BUFFER as short as possible
        let mut w = AUDIO_BUFFER.write().unwrap();
        if w.len() < WINDOW_SIZE {
            return;
        }
        let buffer = w
            .range(..WINDOW_SIZE)
            .map(|value| Complex::new(*value, 0.0))
            .collect::<Vec<_>>();
        w.drain(..HOP_LENGTH);
        buffer
    };

    let mut planner = FftPlanner::new(); // TODO avx planner
    let fft = planner.plan_fft_forward(WINDOW_SIZE);

    // compute the fft
    // TODO use planner with power and abs directly
    fft.process(&mut buffer);

    // compute the power spectrum
    let power: Vec<_> = buffer.into_iter().map(|val| val.norm().powi(2)).collect();

    // compute the mel filterbank
    let n_fft = (WINDOW_SIZE - 1) * 2; // TODO correct?
    let mel_basis = mel_filter::mel::<f32>(
        params.sample_rate as usize,
        n_fft,
        Some(128), // TODO good?
        None,
        None,
        false,
        mel_filter::NormalizationFactor::One,
    );

    // TODO use into_iter everywhere?

    // apply the mel filterbank to the power spectrum
    let mel_power = mel_basis.iter().map(|values| {
        values
            .iter()
            .zip(power.iter())
            .map(|(x, y)| x * y)
            .sum::<f32>()
    });

    // compute the log mel spectrogram
    // TODO  complete this
    // log_mel_power = librosa.power_to_db(mel_power)
    let log_mel_power = mel_power.map(|val| 10. * val.log10());

    // compute the low-frequency amplitude
    low_freq_amp.0 = log_mel_power.take(16).sum();

    // low_freq_amp.0 += 1600.;
    // low_freq_amp.0 *= 0.05;

    dbg!(low_freq_amp.0);
    // dbg!(r.len()); // TODO should not grow
}

fn create_particles(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    low_freq_amp: Res<LowFreqAmp>,
    logo_image: Res<LogoImage>,
    params: Res<Params>,
) {
    // add new particles
    if low_freq_amp.0 > params.particle_amp_threshold {
        let mut rng = rand::thread_rng();
        let theta: f32 = rng.gen_range(0.0..TAU);
        let velocity = Vec2::new(
            params.circle_radius * theta.cos(),
            params.circle_radius * theta.sin(),
        );
        let position = Vec3::from((velocity, 0.0));
        // TODO circle radius is changing!

        if rng.gen_range(0.0..1.0) < params.particle_logo_probability {
            commands.spawn((
                SpriteBundle {
                    texture: logo_image.handle.clone(),
                    transform: Transform::from_translation(position),
                    ..default()
                },
                Particle { velocity },
            ));
        } else {
            let r = rng.gen_range(0.0..0.4);
            let g = rng.gen_range(0.4..1.0);
            let b = rng.gen_range(0.0..0.4);
            commands.spawn((
                MaterialMesh2dBundle {
                    mesh: meshes
                        .add(shape::Circle::new(params.particle_normal_radius).into())
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
    low_freq_amp: Res<LowFreqAmp>,
    params: Res<Params>,
) {
    let window = window_query.get_single().unwrap();
    let max_x = window.width() / 2.;
    let max_y = window.height() / 2.;
    for (entity, particle, mut transform) in &mut query {
        // move particles
        transform.translation.x += particle.velocity.x * params.time_step * low_freq_amp.0;
        transform.translation.y += particle.velocity.y * params.time_step * low_freq_amp.0;

        // remove particles that are off the screen
        if transform.translation.x.abs() > max_x || transform.translation.y.abs() > max_y {
            commands.entity(entity).despawn();
        }
    }
}

// TODO merge with update_circle if they stay the same
fn update_artist_name(
    mut query: Query<&mut Transform, With<ArtistNameText>>,
    low_freq_amp: Res<LowFreqAmp>,
) {
    if let Ok(mut transform) = query.get_single_mut() {
        transform.scale = Vec3::new(low_freq_amp.0, low_freq_amp.0, 0.);
    }
}

fn update_circle(
    mut query: Query<&mut Transform, With<CenterCircle>>,
    low_freq_amp: Res<LowFreqAmp>,
) {
    if let Ok(mut transform) = query.get_single_mut() {
        transform.scale = Vec3::new(low_freq_amp.0, low_freq_amp.0, 0.);
    }
}

use rubato::{InterpolationParameters, InterpolationType, Resampler, SincFixedIn, WindowFunction};

fn init_audio_input(
    device: &Device,
    channels: u16,
    sample_rate: u32,
    buffer_size: u32,
) -> Result<Stream> {
    let supported_input_configs = device.supported_input_configs()?.collect::<Vec<_>>();
    let config = StreamConfig {
        channels,
        sample_rate: SampleRate(sample_rate),
        buffer_size: cpal::BufferSize::Fixed(buffer_size), // TODO what is this doing?
    };

    let channels: usize = channels.into();
    let data_len: usize = channels * N_SAMPLES; // TODO set buffer size to this?
    device
        .build_input_stream(
            &config,
            move |data: &[f32], _: &cpal::InputCallbackInfo| {
                // TODO what if one channel is inverted?

                // assert that data length is an integer multiple of channels, should not be deleted
                assert!(
                    data.len() == data_len,
                    "expected {} but received {}",
                    data_len,
                    data.len()
                );

                // remap to one channel
                // no need to use chunks_exact as exact length is asserted above
                let mut new_samples = [0_f32; N_SAMPLES];
                for (index, samples) in data.chunks(channels).enumerate() {
                    new_samples[index] = samples.iter().sum::<f32>() / (channels as f32);
                }

                // TODO downsample to 22kHz ??

                let params = InterpolationParameters {
                    sinc_len: 256,
                    f_cutoff: 0.95,
                    interpolation: InterpolationType::Linear,
                    oversampling_factor: 256,
                    window: WindowFunction::BlackmanHarris2,
                };
                let mut resampler = SincFixedIn::<f32>::new(
                    sample_rate as f64 / 220500 as f64,
                    2.0,
                    params,
                    N_SAMPLES,
                    1,
                )
                .unwrap();

                let new_samples = vec![new_samples; 1];
                let new_samples = resampler.process(&new_samples, None).unwrap();
                let new_samples = &new_samples[0];
                // dbg!(new_samples.len());

                https://docs.rs/rubato/latest/rubato/




                
                {
                    // lock AUDIO_BUFFER as short as possible
                    let mut w = AUDIO_BUFFER.write().unwrap();
                    if w.len() > AUDIO_BUFFER_MAX_SIZE {
                        println!("AUDIO_BUFFER size exceeded");
                        // remove first N_SAMPLES samples
                        w.drain(..N_SAMPLES);
                    }
                    w.extend(new_samples);
                }
            },
            |err| {
                // TODO change this?
                panic!("{}", err);
            },
            None, // TODO change? Some(Duration::new(2, 0))
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
    params: Res<Params>,
) {
    commands.spawn(Camera2dBundle::default());

    // add center circle
    // TODO particles must spawn below this!
    commands.spawn((
        MaterialMesh2dBundle {
            mesh: meshes
                .add(shape::Circle::new(params.circle_radius).into()) // TODO remove minus
                .into(),
            material: materials.add(ColorMaterial::from(params.circle_color)),
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
                    params.artist_name.clone(),
                    TextStyle {
                        font: asset_server.load("FiraSans-Bold.ttf"),
                        font_size: 60., // TODO font size can be computed?,
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
    // TODO could be loaded into binary and scaled in a build script
    commands.insert_resource(LogoImage {
        handle: asset_server.load("df_logo.png"),
    });
}

// TODO can implement into for String??
fn hex_to_bevy_color(hex: String) -> Result<Color> {
    let (r, g, b) = {
        use colors_transform::{Color, Rgb};
        let color = Rgb::from_hex_str(hex.as_str()).map_err(|err| anyhow!(err.message))?;
        (
            color.get_red() / 255.,
            color.get_green() / 255.,
            color.get_blue() / 255.,
        )
    };
    Ok(Color::rgb(r, g, b))
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
    let params = Params {
        time_step,
        artist_name: args.artist_name,
        particle_amp_threshold: args.particle_amp_threshold,
        particle_logo_probability: args.particle_logo_probability,
        circle_color: hex_to_bevy_color(args.circle_color)?,
        particle_normal_color: hex_to_bevy_color(args.particle_normal_color)?,
        particle_normal_radius: args.particle_normal_radius,
        circle_radius: args.circle_radius,
        sample_rate: args.sample_rate,
    };
    let background_color = hex_to_bevy_color(args.background_color)?;

    // start audio input stream
    let device = devices
        .get(args.input_device_index)
        .ok_or(anyhow!("Invalid input device index"))?;
    let stream = init_audio_input(&device, args.channels, args.sample_rate, args.buffer_size)?;
    stream.play()?;

    App::new()
        // TODO remove default plugins, use only required plugins
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: String::from("DF Visuals"),
                // mode: WindowMode::Fullscreen, // TODO uncomment
                resizable: false,
                resolution: WindowResolution::new(600., 400.), // TODO comment
                window_level: WindowLevel::AlwaysOnTop,
                ..default()
            }),
            ..default()
        }))
        .insert_resource(params)
        .insert_resource(ClearColor(background_color))
        .insert_resource(LowFreqAmp(0.))
        .add_startup_system(setup)
        .add_systems(
            (
                get_low_freq_amp,
                update_artist_name.after(get_low_freq_amp),
                update_circle.after(get_low_freq_amp),
                create_particles.after(get_low_freq_amp),
                update_particles.after(create_particles),
            )
                .in_schedule(CoreSchedule::FixedUpdate),
        )
        .insert_resource(FixedTime::new_from_secs(time_step))
        .add_system(bevy::window::close_on_esc)
        .run();

    Ok(())
}
