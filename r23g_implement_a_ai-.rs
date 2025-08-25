use std::collections::{HashMap, VecDeque};
use std::fs;
use std::io::{self, prelude::*};
use std::path::Path;

extern crate rustlearn;
extern crate csv;

use rustlearn::LinearRegression;
use csv::Reader;

struct Pipeline {
    data: VecDeque<String>,
    models: HashMap<String, LinearRegression>,
}

impl Pipeline {
    fn new() -> Self {
        Pipeline {
            data: VecDeque::new(),
            models: HashMap::new(),
        }
    }

    fn ingest(&mut self, file_path: &str) -> io::Result<()> {
        let file = fs::File::open(file_path)?;
        let reader = Reader::from_reader(file);
        for result in reader.records() {
            let record = result?;
            self.data.push_back(record.join(","));
        }
        Ok(())
    }

    fn train_model(&mut self, name: &str, features: Vec<usize>) -> io::Result<()> {
        let mut x = vec![];
        let mut y = vec![];
        for row in &self.data {
            let columns: Vec<&str> = row.split(",").collect();
            let mut feature_row = vec![];
            for &feature in &features {
                feature_row.push(columns[feature].parse::<f64>().unwrap());
            }
            x.push(feature_row);
            y.push(columns.last().unwrap().parse::<f64>().unwrap());
        }
        let mut model = LinearRegression::new();
        model.train(&x, &y)?;
        self.models.insert(name.to_string(), model);
        Ok(())
    }

    fn predict(&self, model_name: &str, input: Vec<f64>) -> f64 {
        let model = self.models.get(model_name).unwrap();
        model.predict(&input)
    }
}

fn main() -> io::Result<()> {
    let mut pipeline = Pipeline::new();
    pipeline.ingest("data.csv")?;
    pipeline.train_model("model1", vec![1, 2, 3])?;
    let prediction = pipeline.predict("model1", vec![1.0, 2.0, 3.0]);
    println!("Prediction: {}", prediction);
    Ok(())
}