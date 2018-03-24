extern crate rand;
extern crate ndarray;
extern crate ndarray_rand;
extern crate ndarray_parallel;
extern crate rayon;
extern crate itertools;
extern crate num_traits;

use itertools::Zip;
use itertools::Itertools;
use std::ops::{Add, Sub, Mul, Div};
use rand::distributions::*;
use rand::Rng;
use ndarray::prelude::*;
use ndarray::{
    RemoveAxis,
    Array1,
    Array2,
    Array3,
    arr1,
};
use ndarray_rand::RandomExt;
use ndarray_parallel::prelude::*;
use ndarray_parallel::*;
use std::ptr::copy_nonoverlapping;
use std::ptr::replace;
use num_traits::Zero;
use lbm::*;

const FOUR9THS: f64 = 4.0 / 9.0;
const ONE9THS: f64 = 1.0 / 9.0;
const ONE36THS: f64 = 1.0 / 36.0;
const WEIGHTS: [f64; 9] = [FOUR9THS, ONE9THS, ONE9THS, ONE9THS, ONE9THS, ONE36THS, ONE36THS, ONE36THS, ONE36THS];
const REVERSE: [usize; 9] = [0, 2, 1, 4, 3, 8, 7, 6, 5];
const FVEC_E: [[f64; 2]; 9] = [[0.0, 0.0], [1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]];
const IVEC_E: [[isize; 2]; 9] = [[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];
const DT: f64 = 1.0;
const VISCOSITY: f64 = 0.08;

const FARR_E1: [f64; 9] = [0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0];
const FARR_E2: [f64; 9] = [0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];

pub struct LBMState2 {
    width: usize,
    height: usize,
    mag: usize,
    omega: f64,
    red: Array3<f64>,
    blue: Array3<f64>,
    rho_r: Array2<f64>,
    rho_b: Array2<f64>,
    ux: Array2<f64>,
    uy: Array2<f64>,
    block: Array3<bool>,
}

impl LBMState2 {
    pub fn new(w: usize, h: usize, m: usize) -> LBMState2 {
        let wm = w / m;
        let hm = h / m;
        let size = wm * hm;
        LBMState2 {
            width: wm,
            height: hm,
            mag: m,
            omega: 1.0 / (3.0 * VISCOSITY / DT + 0.5),
            red: Array::random((9, wm, hm), Range::new(0., 1. / 9.)),//WEIGHTS.iter().map(|a| { Array::random((wm, hm), Range::new(0., *a)) }).collect(),
            blue: Array::random((9, wm, hm), Range::new(0., 1. / 9.)),//WEIGHTS.iter().map(|a| { Array::random((wm, hm), Range::new(0., *a)) }).collect(),
            rho_r: Array::random((wm, hm), Range::new(0., 1.)),
            rho_b: Array::random((wm, hm), Range::new(0., 1.)),
            ux: Array::<f64, _>::zeros((wm, hm)),
            uy: Array::<f64, _>::zeros((wm, hm)),
            block: Array::<bool, _>::from_vec(vec![false; size * 9]).into_shape((9, wm, hm)).unwrap(),
        }
    }

    pub fn stream(&mut self) {
        flow(&mut self.red, &self.block, IVEC_E);
        flow(&mut self.blue, &self.block, IVEC_E);
    }

    pub fn update_state(&mut self) {
        ndarray::Zip::from(self.rho_r.view_mut())
            .and(self.rho_b.view_mut())
            .and(self.ux.view_mut())
            .and(self.uy.view_mut())
            .and(self.red.gencolumns())
            .and(self.blue.gencolumns())
            .par_apply(|pr, pb, x, y, rs: ArrayView1<f64>, bs: ArrayView1<f64>| {
                *pr = rs.scalar_sum();
                *pb = bs.scalar_sum();
                let x1: f64 = FARR_E1.iter().zip(rs.iter()).map(|(a, b)| { a * b }).sum();
                let y1: f64 = FARR_E2.iter().zip(rs.iter()).map(|(a, b)| { a * b }).sum();
                let x2: f64 = FARR_E1.iter().zip(bs.iter()).map(|(a, b)| { a * b }).sum();
                let y2: f64 = FARR_E2.iter().zip(bs.iter()).map(|(a, b)| { a * b }).sum();
                *x = (x1 + x2) / (*pr + *pb);
                *y = (y1 + y2) / (*pr + *pb);
            });
    }

    pub fn collide(&mut self) {
        for (i, (r, b)) in self.red.outer_iter_mut().zip(self.blue.outer_iter_mut()).enumerate() {
            process(r, b, &self.rho_r, &self.rho_b, &self.ux, &self.uy, self.omega, i);
        }
        fn process(r: ArrayViewMut2<f64>, b: ArrayViewMut2<f64>, rho_r: &Array2<f64>, rho_b: &Array2<f64>, ux: &Array2<f64>, uy: &Array2<f64>, omg: f64, i: usize) {
            ndarray::Zip::from(r)
                .and(b)
                .and(rho_r)
                .and(rho_b)
                .and(ux)
                .and(uy)
                .apply(|red, blue, &p_r, &p_b, &x, &y| {
                    let k = 0.0;
                    let v_r = x * x + (y + k) * (y + k);//for iの外に出せる
                    let v_b = x * x + (y - k) * (y - k);//for iの外に出せる
                    let e_dir_r = x * FVEC_E[i][0] + (y + k) * FVEC_E[i][1];
                    let e_dir_b = x * FVEC_E[i][0] + (y - k) * FVEC_E[i][1];
                    *red = (1.0 - omg) * *red + p_r * omg * WEIGHTS[i] * (1.0 - 1.5 * v_r + 3.0 * e_dir_r + 4.5 * e_dir_r * e_dir_r);
                    *blue = (1.0 - omg) * *blue + p_b * omg * WEIGHTS[i] * (1.0 - 1.5 * v_b + 3.0 * e_dir_b + 4.5 * e_dir_b * e_dir_b);
                });
        }
    }

    fn force(&mut self) {
        ndarray::Zip::from(self.rho_r.view())
            .and(self.rho_b.view())
            .and(self.ux.view_mut())
            .and(self.uy.view_mut())
            .par_apply(|pr, pb, x, y|{
                *y = *y + pr * 0.001;
            })
    }
}

impl Fluid for LBMState2 {
    fn init(&mut self) {
        let v = WEIGHTS.iter().map(|x| { vec![*x * 0.3; self.width * self.height] }).flatten().collect_vec();
        self.red = Array::from_vec(v).into_shape((9, self.width, self.height)).unwrap();
        self.red.slice_mut(s![.., ..self.height / 2, ..]).fill(0.0);

        let v = WEIGHTS.iter().map(|x| { vec![*x * 0.3; self.width * self.height] }).flatten().collect_vec();
        self.blue = Array::from_vec(v).into_shape((9, self.width, self.height)).unwrap();
        self.blue.slice_mut(s![.., self.height / 2.., ..]).fill(0.0);
        //Array::random((9, self.width, self.height), Range::new(0., 1. / 9.));//

        let mut block = Array::<bool, _>::from_elem((self.width, self.height), false);
        block.slice_mut(s![50..51, self.height / 2 - 8..self.height / 2 + 8]).fill(true);
        block.slice_mut(s![.., ..1]).fill(true);
        block.slice_mut(s![.., -1..]).fill(true);
        for i in 0..9 {
            let c = roll(&block.view(), &IVEC_E[i], false);
            self.block.subview_mut(Axis(0), i).assign(&c);
        };
    }

    fn draw(&self, pixels: &mut Vec<u8>) {
        for row in 0..self.height {
            for col in 0..self.width {
                if self.block[[0, col, row]] {
                    for x in 0..self.mag {
                        for y in 0..self.mag {
                            pixels[((((row * self.mag + y) * self.width + col) * self.mag + x) * 4) as usize] = 255;
                            pixels[((((row * self.mag + y) * self.width + col) * self.mag + x) * 4 + 1) as usize] = 255;
                            pixels[((((row * self.mag + y) * self.width + col) * self.mag + x) * 4 + 2) as usize] = 255;
                        }
                    }
                    continue
                }
                let r = adjust(self.rho_r[[col, row]], 0.35, 0.0);
                let b = adjust(self.rho_b[[col, row]], 0.35, 0.0);
                for x in 0..self.mag {
                    for y in 0..self.mag {
                        pixels[((((row * self.mag + y) * self.width + col) * self.mag + x) * 4) as usize] = r;
                        pixels[((((row * self.mag + y) * self.width + col) * self.mag + x) * 4 + 2) as usize] = b;
                    }
                }
            }
        }

        fn adjust(x: f64, max: f64, min: f64) -> u8 {
            ((x.max(min).min(max) - min) * 255.0 / (max - min)) as u8
        }
    }

    fn update(&mut self) {
        self.stream();
        self.update_state();
        self.force();
        self.collide();
    }

    fn left_click(&mut self, x: f64, y: f64) {
        let r = 10;
        for xi in -r..r {
            for yi in -r..r {
                for i in 0..9 {
                    let x1 = ((x as i32 / self.mag as i32) + xi) as usize;
                    let y1 = ((y as i32 / self.mag as i32) + yi) as usize;
                    if 0 <= x1 && x1 < self.width && 0 <= y1 && y1 < self.height {
                        self.red[[i, x1, y1]] += WEIGHTS[i] * 0.8 * ((-xi * xi + -yi * yi) as f64 / 1.0).exp();
                    }
                }
            }
        }
    }

    fn right_click(&mut self, x: f64, y: f64) {
        let r = 10;
        for xi in -r..r {
            for yi in -r..r {
                for i in 0..9 {
                    let x1 = ((x as i32 / self.mag as i32) + xi) as usize;
                    let y1 = ((y as i32 / self.mag as i32) + yi) as usize;
                    if 0 <= x1 && x1 < self.width && 0 <= y1 && y1 < self.height {
                        if r * r > (xi * xi + yi * yi) {
                            self.red[[i, x1, y1]] -= WEIGHTS[i] * 0.1;
                        }
                    }
                }
            }
        }
    }
}