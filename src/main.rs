use bevy::{
    prelude::*,
    sprite::{Anchor, MaterialMesh2dBundle},
    text::{BreakLineOn, Text2dBounds},
    window::{WindowLevel, WindowMode, WindowResolution},
};
use rand::Rng;
use std::f32::consts::TAU;

const CENTER: Vec3 = Vec3::new(0.0, 0.0, 0.0);

const ARTIST_NAME: &str = "Ar ti st Name";
const ARTIST_NAME_FONT_SIZE: f32 = 60.; // TODO can be computed?

const TIME_STEP: f32 = 1.0 / 35.0;
const BACKGROUND_COLOR: Color = Color::rgb(0.9, 0.9, 0.9);

const PARTICLE_NORMAL_COLOR: Color = Color::rgb(1.0, 0.5, 0.5);
const PARTICLE_NORMAL_RADIUS: f32 = 10.0;

const PARTICLE_LOGO_PROBABILITY: f32 = 0.2;
const PARTICLE_DF_RADIUS: f32 = 20.0;

const CIRCLE_COLOR: Color = Color::rgb(0.5, 0.5, 1.0);
const CIRCLE_RADIUS: f32 = 100.0;

#[derive(Component)]
struct Particle {
    velocity: Vec2,
}

#[derive(Component)]
struct ArtistNameText;

#[derive(Resource)]
struct LogoImage {
    handle: Handle<Image>,
}

fn fetch_audio_samples(asset_server: Res<AssetServer>) {
    // TODO
}

fn create_particles(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    logo_image: Res<LogoImage>,
) {
    let mut rng = rand::thread_rng();

    if rng.gen_range(0.0..1.0) > 0.2 {
        return;
    }

    // add new particles
    let theta: f32 = rng.gen_range(0.0..TAU);
    let velocity = Vec2::new(CIRCLE_RADIUS * theta.cos(), CIRCLE_RADIUS * theta.sin());
    let position = CENTER + Vec3::from((velocity, 0.0));

    if rng.gen_range(0.0..1.0) < PARTICLE_LOGO_PROBABILITY {
        commands.spawn((
            SpriteBundle {
                texture: logo_image.handle.clone(),
                transform: Transform::from_translation(position),
                ..default()
            },
            Particle { velocity },
        ));
    } else {
        commands.spawn((
            MaterialMesh2dBundle {
                mesh: meshes
                    .add(shape::Circle::new(PARTICLE_NORMAL_RADIUS).into())
                    .into(),
                material: materials.add(ColorMaterial::from(PARTICLE_NORMAL_COLOR)),
                transform: Transform::from_translation(position),
                ..default()
            },
            Particle { velocity },
        ));
    }
}

fn update_particles(
    mut commands: Commands,
    mut query: Query<(Entity, &Particle, &mut Transform)>,
    window_query: Query<&Window>,
) {
    let window = window_query.get_single().unwrap();
    let max_x = window.width() / 2.;
    let max_y = window.height() / 2.;
    for (entity, particle, mut transform) in &mut query {
        // move particles
        transform.translation.x += particle.velocity.x * TIME_STEP;
        transform.translation.y += particle.velocity.y * TIME_STEP;

        // remove particles that are off the screen
        if transform.translation.x.abs() > max_x || transform.translation.y.abs() > max_y {
            println!("despawn {:?}", entity);
            commands.entity(entity).despawn();
        }
    }
}

fn update_artist_name(mut query: Query<&mut Text, With<ArtistNameText>>) {
    // TODO
    // for mut text in &mut query {
    //     text.sections[0].style.font_size = if text.sections[0].style.font_size > 100. {
    //         50.
    //     } else {
    //         150.
    //     };
    // }
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    asset_server: Res<AssetServer>,
) {
    commands.spawn(Camera2dBundle::default());

    // add center circle
    // TODO particles must spawn below this!
    commands.spawn(MaterialMesh2dBundle {
        mesh: meshes
            .add(shape::Circle::new(CIRCLE_RADIUS).into()) // TODO remove minus
            .into(),
        material: materials.add(ColorMaterial::from(CIRCLE_COLOR)),
        transform: Transform::from_translation(CENTER),
        ..default()
    });

    // add artist name text
    commands.spawn((
        Text2dBundle {
            text: Text {
                sections: vec![TextSection::new(
                    ARTIST_NAME,
                    TextStyle {
                        font: asset_server.load("FiraSans-Bold.ttf"),
                        font_size: ARTIST_NAME_FONT_SIZE,
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

fn main() {
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
        .insert_resource(ClearColor(BACKGROUND_COLOR))
        .add_startup_system(setup)
        .add_systems(
            (
                fetch_audio_samples,
                update_artist_name.after(fetch_audio_samples),
                create_particles.after(fetch_audio_samples),
                update_particles.after(create_particles),
            )
                .in_schedule(CoreSchedule::FixedUpdate),
        )
        .insert_resource(FixedTime::new_from_secs(TIME_STEP))
        .add_system(bevy::window::close_on_esc)
        .run();
}
