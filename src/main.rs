extern crate piston_window;
extern crate image as im;
extern crate fluid_simulation;
extern crate rand;

use piston_window::*;
use fluid_simulation::lbm::*;
use std::time::{Duration, Instant};

//use rand::Rng;

const WIDTH: u32 = 512;
const HEIGHT: u32 = 512;
const RATIO: u32 = 4;

fn main() {
    let opengl = OpenGL::V3_2;
    let (width, height) = (WIDTH, HEIGHT);
    let mut window: PistonWindow =
        WindowSettings::new("fluid simulation", (width, height))
            .exit_on_esc(true)
            .opengl(opengl)
            .build()
            .unwrap();

    let mut draw = false;
    let mut texture: G2dTexture = Texture::from_image(
        &mut window.factory,
        &im::ImageBuffer::new(width, height),
        &TextureSettings::new(),
    ).unwrap();

    //let mut rng = rand::thread_rng();
    //let mut f = [rng.gen_range(0., 1.); (WIDTH * HEIGHT) as usize];
    let mut pixels = vec![[0u8, 0u8, 0u8, 255u8]; (WIDTH * HEIGHT) as usize].concat();

    let mut fluid = LBMState2s::new(width as usize, height as usize, RATIO as usize);
    fluid.init();

    let mut x = 0.0;
    let mut y = 0.0;

    {
        /*fluid.update();
        let start = Instant::now();
        for _i in 0..1000 {
            fluid.collide();
        }
        let end = start.elapsed();
        println!("collide {}.{:03}", end.as_secs(), end.subsec_nanos() / 1_000_000);

        let start = Instant::now();
        for _i in 0..1000 {
            fluid.update_state();
        }
        let end = start.elapsed();
        println!("update_macro {}.{:03}", end.as_secs(), end.subsec_nanos() / 1_000_000);

        let start = Instant::now();
        for _i in 0..1000 {
            fluid.stream();
        }
        let end = start.elapsed();
        println!("stream {}.{:03}", end.as_secs(), end.subsec_nanos() / 1_000_000);*/
    }

    /*let mut frames = 0;
    let mut passed = 0.0;
    let mut start = Instant::now();*/

    while let Some(e) = window.next() {
        if let Some(_) = e.render_args() {
            fluid.draw(&mut pixels);

            let image0: im::ImageBuffer<im::Rgba<u8>, Vec<u8>> = im::ImageBuffer::from_raw(width, height, pixels.clone()).unwrap();
            texture.update(&mut window.encoder, &image0).unwrap();
            window.draw_2d(&e, |c, g| {
                clear([1.0; 4], g);
                image(&texture, c.transform, g);
            });
            /*frames += 1;
            if frames > 60 {
                frames = 0;
                let end = start.elapsed();
                println!("{}.{:03}", end.as_secs(), end.subsec_nanos() / 1_000_000);
                start = Instant::now();
            }*/
        };
        if let Some(_) = e.update_args() {
            for _i in 0..4 {
                fluid.update();
            }
            if draw {
                fluid.left_click(x, y);
            }
        };
        if let Some(button) = e.press_args() {
            if button == Button::Mouse(MouseButton::Left) {
                draw = true;
            } else if button == Button::Mouse(MouseButton::Right) {
                fluid.right_click(x, y);
            } else if button == Button::Keyboard(Key::R) {
                fluid.init();
            }
        };
        if let Some(button) = e.release_args() {
            if button == Button::Mouse(MouseButton::Left) {
                draw = false;
            }
        };
        if let Some(pos) = e.mouse_cursor_args() {
            x = pos[0];
            y = pos[1];
        };
    }
}
