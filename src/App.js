import React, { useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as Papa from 'papaparse';

const GanadoPredictivo = () => {
    const [predicciones, setPredicciones] = useState([]);

    // Función para cargar y preprocesar los datos de entrenamiento
    const loadData = async (filePath) => {
        const response = await fetch(filePath);
        const data = await response.text();

        const parsedData = Papa.parse(data, { header: true }).data;

        const inputs = [];
        const labels = [];

        parsedData.forEach(row => {
            const lat = parseFloat(row.Latitud);
            const long = parseFloat(row.Longitud);
            const velocidad = parseFloat(row.Velocidad);
            const comportamiento = row.Comportamiento === 'Normal' ? 0 : 1;

            inputs.push([lat, long, velocidad]);
            labels.push([comportamiento]);
        });

        return {
            inputs: tf.tensor2d(inputs),
            labels: tf.tensor2d(labels),
        };
    };

    // Función para crear y entrenar el modelo
    const createAndTrainModel = async () => {
        const { inputs, labels } = await loadData('/comportamiento_ganado_simulado.csv'); // Ruta del archivo CSV de entrenamiento
        const model = await createModel();
        await trainModel(model, inputs, labels);
        
        // Después de entrenar, cargar los datos de prueba y hacer predicciones
        const { inputs: testInputs } = await loadData('/comportamiento_ganado_simulado_prueba.csv'); // Ruta del archivo CSV de prueba
        makePredictions(model, testInputs);
    };

    const createModel = async () => {
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 10, activation: 'relu', inputShape: [3] }));
        model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

        model.compile({ optimizer: 'adam', loss: 'binaryCrossentropy', metrics: ['accuracy'] });

        return model;
    };

    const trainModel = async (model, inputs, labels) => {
        await model.fit(inputs, labels, {
            epochs: 100,
            batchSize: 32,
            validationSplit: 0.2,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch: ${epoch}, Loss: ${logs.loss}, Accuracy: ${logs.acc}`);
                },
            },
        });
    };

    const makePredictions = async (model, inputs) => {
        const prediccionesTensor = model.predict(inputs);
        const prediccionesArray = await prediccionesTensor.array();

        // Convertir las predicciones a etiquetas
        const resultados = prediccionesArray.map(pred => (pred[0] > 0.5 ? 'Desviado' : 'Normal'));
        setPredicciones(resultados);
    };

    useEffect(() => {
        createAndTrainModel();
    }, []);

    return (
        <div>
            <h1>Entrenando el modelo...</h1>
            <h2>Predicciones de Comportamiento:</h2>
            <ul>
                {predicciones.map((pred, index) => (
                    <li key={index}>{pred}</li>
                ))}
            </ul>
        </div>
    );
};

export default GanadoPredictivo;
