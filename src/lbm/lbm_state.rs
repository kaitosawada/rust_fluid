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
const VISCOSITY: f64 = 0.05;

const FARR_E1: [f64; 9] = [0.0, 1.0, -1.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0];
const FARR_E2: [f64; 9] = [0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0];

pub struct LBMState {
    width: usize,
    height: usize,
    mag: usize,
    omega: f64,
    ingredient: Array3<f64>,
    rho: Array2<f64>,
    ux: Array2<f64>,
    uy: Array2<f64>,
    block: Array3<bool>,
}

impl LBMState {
    pub fn new(w: usize, h: usize, m: usize) -> LBMState {
        let wm = w / m;
        let hm = h / m;
        let size = wm * hm;
        LBMState {
            width: wm,
            height: hm,
            mag: m,
            omega: 1.0 / (3.0 * VISCOSITY / DT + 0.5),
            ingredient: Array::random((9, wm, hm), Range::new(0., 1. / 9.)),//WEIGHTS.iter().map(|a| { Array::random((wm, hm), Range::new(0., *a)) }).collect(),
            rho: Array::random((wm, hm), Range::new(0., 1.)),
            ux: Array::<f64, _>::zeros((wm, hm)),
            uy: Array::<f64, _>::zeros((wm, hm)),
            block: Array::<bool, _>::from_vec(vec![false; size * 9]).into_shape((9, wm, hm)).unwrap(),
        }
    }

    pub fn stream(&mut self) {
        flow(&mut self.ingredient, &self.block, IVEC_E);
    }

    pub fn update_state(&mut self) {
        ndarray::Zip::from(self.rho.view_mut())
            .and(self.ux.view_mut())
            .and(self.uy.view_mut())
            .and(self.ingredient.gencolumns())
            .par_apply(|p, x, y, cs: ArrayView1<f64>| {
                *p = cs.scalar_sum();
                *x = FARR_E1.iter().zip(cs.iter()).map(|(a, b)| { a * b }).sum();
                *y = FARR_E2.iter().zip(cs.iter()).map(|(a, b)| { a * b }).sum();
                *x = *x / *p;
                *y = *y / *p;
            });
    }

    pub fn collide(&mut self) {
        for (i, item) in self.ingredient.outer_iter_mut().enumerate() {
            process(item, &self.rho, &self.ux, &self.uy, self.omega, i);
        }
        fn process(ingr: ArrayViewMut2<f64>, rho: &Array2<f64>, ux: &Array2<f64>, uy: &Array2<f64>, omg: f64, i: usize) {
            ndarray::Zip::from(ingr)
                .and(rho)
                .and(ux)
                .and(uy)
                .apply(|c, &p, &x, &y| {
                    let v = x * x + y * y;//for iの外に出せる
                    let e_dir = x * FVEC_E[i][0] + y * FVEC_E[i][1];
                    *c = (1.0 - omg) * *c + p * omg * WEIGHTS[i] * (1.0 - 1.5 * v + 3.0 * e_dir + 4.5 * e_dir * e_dir);
                });
        }
    }

    fn force(&mut self) {
        ndarray::Zip::from(self.ux.subview_mut(Axis(0), 2))
            .and(self.uy.subview_mut(Axis(0), 0))
            .par_apply(|x, y| {
                *x += 0.01;
            })
        //self.ux.slice_mut(s![..5, ..]).fill(0.1);
    }
}

impl Fluid for LBMState {
    fn init(&mut self) {
        let v = WEIGHTS.iter().map(|x| { vec![*x * 0.3; self.width * self.height] }).flatten().collect_vec();
        self.ingredient = Array::from_vec(v).into_shape((9, self.width, self.height)).unwrap();
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
                let u = adjust(self.rho[[col, row]], 0.35, 0.25);
                for x in 0..self.mag {
                    for y in 0..self.mag {
                        pixels[((((row * self.mag + y) * self.width + col) * self.mag + x) * 4 + 1) as usize] = u;
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
        //self.force();
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
                        self.ingredient[[i, x1, y1]] += WEIGHTS[i] * 0.8 * ((-xi * xi + -yi * yi) as f64 / 1.0).exp();
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
                            self.ingredient[[i, x1, y1]] -= WEIGHTS[i] * 0.1;
                        }
                    }
                }
            }
        }
    }
}