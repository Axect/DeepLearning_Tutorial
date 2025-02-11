use peroxide::fuga::*;
use dialoguer::{Select, theme::ColorfulTheme};
use rugfield::{grf_with_rng, Kernel};

const SEED: u64 = 125;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let n = 3000;

    let (df, dg) = create_regression_data(n)?;
    println!("True Data");
    df.print();
    df.write_parquet("true.parquet", CompressionOptions::Snappy)?;
    println!("Train & Validation Data");
    dg.print();
    dg.write_parquet("data.parquet", CompressionOptions::Snappy)?;

    Ok(())
}

fn create_regression_data(n: usize) -> Result<(DataFrame, DataFrame), Box<dyn std::error::Error>> {
    let mut rng = stdrng_from_seed(SEED);

    let kernel = Kernel::SquaredExponential(0.15);
    let grf_data = grf_with_rng(&mut rng, n, kernel);

    let x = linspace(0, 1, n);

    let cs = cubic_hermite_spline(&x, &grf_data, Quadratic)?;

    let uniform = Uniform(0.0, 1.0);
    let normal = Normal(0.0, 0.1);
    let x_new = uniform.sample_with_rng(&mut rng, n);
    let y_true = cs.eval_vec(&x_new);
    let noise = normal.sample_with_rng(&mut rng, n);
    let y_new = y_true.add_vec(&noise);

    let mut df = DataFrame::new(vec![]);
    df.push("x", Series::new(x));
    df.push("y", Series::new(grf_data));
    
    let mut dg = DataFrame::new(vec![]);
    dg.push("x", Series::new(x_new));
    dg.push("y", Series::new(y_new));

    Ok((df, dg))
}
