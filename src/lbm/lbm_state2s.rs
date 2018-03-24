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

pub struct LBMState2s {
    width: usize,
    height: usize,
    mag: usize,
    omega: f64,
    liq: Array3<f64>,
    gas: Array3<f64>,
    rho_l: Array2<f64>,
    rho_g: Array2<f64>,
    ux: Array2<f64>,
    uy: Array2<f64>,
    ux_g: Array2<f64>,
    uy_g: Array2<f64>,
    block: Array3<bool>,
}

impl LBMState2s {
    pub fn new(w: usize, h: usize, m: usize) -> LBMState2s {
        let wm = w / m;
        let hm = h / m;
        let size = wm * hm;
        LBMState2s {
            width: wm,
            height: hm,
            mag: m,
            omega: 1.0 / (3.0 * VISCOSITY / DT + 0.5),
            liq: Array::random((9, wm, hm), Range::new(0., 1. / 9.)),//WEIGHTS.iter().map(|a| { Array::random((wm, hm), Range::new(0., *a)) }).collect(),
            gas: Array::random((9, wm, hm), Range::new(0., 1. / 9.)),//WEIGHTS.iter().map(|a| { Array::random((wm, hm), Range::new(0., *a)) }).collect(),
            rho_l: Array::random((wm, hm), Range::new(0., 1.)),
            rho_g: Array::random((wm, hm), Range::new(0., 1.)),
            ux: Array::<f64, _>::zeros((wm, hm)),
            uy: Array::<f64, _>::zeros((wm, hm)),
            ux_g: Array::<f64, _>::zeros((wm, hm)),
            uy_g: Array::<f64, _>::zeros((wm, hm)),
            block: Array::<bool, _>::from_vec(vec![false; size * 9]).into_shape((9, wm, hm)).unwrap(),
        }
    }

    pub fn stream(&mut self) {
        flow(&mut self.liq, &self.block, IVEC_E);
        flow(&mut self.gas, &self.block, IVEC_E);
    }

    pub fn update_state(&mut self) {
        ndarray::Zip::from(self.rho_l.view_mut())
            .and(self.rho_g.view_mut())
            .and(self.ux.view_mut())
            .and(self.uy.view_mut())
            .and(self.liq.gencolumns())
            .and(self.gas.gencolumns())
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
        self.ux_g.assign(&self.ux);
        self.uy_g.assign(&self.uy);
    }

    pub fn collide(&mut self) {
        for (i, (l, g)) in self.liq.outer_iter_mut().zip(self.gas.outer_iter_mut()).enumerate() {
            process(l, g, &self.rho_l, &self.rho_g, &self.ux, &self.uy, &self.ux_g, &self.uy_g, self.omega, i);
        }
        fn process(l: ArrayViewMut2<f64>, g: ArrayViewMut2<f64>, rho_l: &Array2<f64>, rho_g: &Array2<f64>, ux: &Array2<f64>, uy: &Array2<f64>, ux_g: &Array2<f64>, uy_g: &Array2<f64>, omg: f64, i: usize) {
            ndarray::Zip::from(l)
                .and(rho_l)
                .and(ux)
                .and(uy)
                .apply(|l, &p_l, &x, &y| {
                    let v_l = x * x + y * y;
                    let e_dir_l = x * FVEC_E[i][0] + y * FVEC_E[i][1];
                    *l = (1.0 - omg) * *l + p_l * omg * WEIGHTS[i] * (1.0 - 1.5 * v_l + 3.0 * e_dir_l + 4.5 * e_dir_l * e_dir_l);
                });
            ndarray::Zip::from(g)
                .and(rho_g)
                .and(ux_g)
                .and(uy_g)
                .apply(|g, &p_g, &x_g, &y_g| {
                    let v_g = x_g * x_g + y_g * y_g;
                    let e_dir_g = x_g * FVEC_E[i][0] + y_g * FVEC_E[i][1];
                    *g = (1.0 - omg) * *g + p_g * omg * WEIGHTS[i] * (1.0 - 1.5 * v_g + 3.0 * e_dir_g + 4.5 * e_dir_g * e_dir_g);
                });
        }
    }

    fn force(&mut self) {
        let dim = self.ux.raw_dim();
        let kappa = 2.0;
        let phi = 0.005;
        let dx = 1.0;
        let mut rho = self.rho_l.view().add(&self.rho_g);
        let mut ux = self.ux.view().add(&self.ux_g);
        let mut uy = self.uy.view().add(&self.uy_g);
        let mut p = rho.view().div(3.0);
        let rho_l_ref = 0.3;
        let beta = 5000.0;
        let mut p_ = Array::zeros(dim);
        ndarray::Zip::from(&mut p_)
            .and(&p)
            .and(&self.rho_l)
            .apply(|p_, p, rho| {
                *p_ = p + beta * (rho - rho_l_ref);
            });
        let (nab_rho_l_x, nab_rho_l_y) = nabla(&self.rho_l.view(), dx);
        let s_l_x = self.rho_g.view().mul(&nab_rho_l_x.view()).mul(kappa);
        let s_l_y = self.rho_g.view().mul(&nab_rho_l_y.view()).mul(kappa);
        let s_g_x = self.rho_l.view().mul(&nab_rho_l_x.view()).mul(-kappa);
        let s_g_y = self.rho_l.view().mul(&nab_rho_l_y.view()).mul(-kappa);
        let ddu_x = lapla(&ux.view(), dx);
        let ddu_y = lapla(&uy.view(), dx);
        let (dpx, dpy) = nabla(&p.view(), dx);
        let (dp_x, dp_y) = nabla(&p_.view(), dx);
        let mut ax = Array::zeros(p.dim());
        let mut ay = Array::zeros(p.dim());
        ndarray::Zip::from(&mut ax)
            .and(&rho)
            .and(&dpx)
            .and(&dp_x)
            .and(&ddu_x)
            .par_apply(|ax, &rho, &dp, &dp_, &ddu| {
                if rho == 0.0 {
                    *ax = -ddu * VISCOSITY + 1.0 / 15.0 * ddu * VISCOSITY;
                } else {
                    *ax = dp / rho - ddu * VISCOSITY - 1.0 / 800.0 * dp_ / rho + 1.0 / 15.0 * ddu * VISCOSITY;
                }
            });
        ndarray::Zip::from(&mut ay)
            .and(&rho)
            .and(&dpy)
            .and(&dp_y)
            .and(&ddu_y)
            .par_apply(|ay, &rho, &dp, &dp_, &ddu| {
                if rho == 0.0 {
                    *ay = -ddu * VISCOSITY + 1.0 / 15.0 * ddu * VISCOSITY;
                } else {
                    *ay = dp / rho - ddu * VISCOSITY - 1.0 / 800.0 * dp_ / rho + 1.0 / 15.0 * ddu * VISCOSITY;
                }
            });
        let g = 0.0001;
        ndarray::Zip::from(self.ux.view_mut())
            .and(self.uy.view_mut())
            .and(&s_l_x)
            .and(&s_l_y)
            .and(&ax)
            .and(&ay)
            .par_apply(|ux, uy, &slx, &sly, &ax, &ay| {
                *ux += slx + ax * phi;
                *uy += sly + g + ay * phi;
            });

        ndarray::Zip::from(self.ux_g.view_mut())
            .and(self.uy_g.view_mut())
            .and(&s_g_x)
            .and(&s_g_y)
            .par_apply(|ux, uy, &slx, &sly| {
                *ux += slx;
                *uy += sly;
            });
    }
}

impl Fluid for LBMState2s {
    fn init(&mut self) {
        let v = WEIGHTS.iter().map(|x| { vec![*x * 0.15; self.width * self.height] }).flatten().collect_vec();
        self.liq = Array::from_vec(v).into_shape((9, self.width, self.height)).unwrap();
        //self.liq.slice_mut(s![.., ..self.height / 2, ..]).fill(0.01);

        let v = WEIGHTS.iter().map(|x| { vec![*x * 0.3; self.width * self.height] }).flatten().collect_vec();
        self.gas = Array::from_vec(v).into_shape((9, self.width, self.height)).unwrap();
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
                let r = adjust(self.rho_l[[col, row]], 0.35, 0.0);
                let b = adjust(self.rho_g[[col, row]], 0.35, 0.0);
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
                        self.liq[[i, x1, y1]] += WEIGHTS[i] * 0.8 * ((-xi * xi + -yi * yi) as f64 / 1.0).exp();
                    }
                }
            }
        }
    }

    fn right_click(&mut self, x: f64, y: f64) {
        let x1 = (x as i32 / self.mag as i32) as usize;
        let y1 = (y as i32 / self.mag as i32) as usize;
        if 0 <= x1 && x1 < self.width && 0 <= y1 && y1 < self.height {
            println!("{}", self.rho_l[[x1, y1]]);
        }
    }
}