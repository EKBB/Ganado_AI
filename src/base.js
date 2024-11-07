import React, { useEffect, useState } from 'react';
import * as tf from '@tensorflow/tfjs';
import * as Papa from 'papaparse';

const GanadoPredictivo = () => {
    const [predicciones, setPredicciones] = useState([]);

    // Función para cargar y normalizar los datos
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

            if (!isNaN(lat) && !isNaN(long) && !isNaN(velocidad)) {
                // Normalizar las entradas dividiendo por el máximo esperado (ajuste según sea necesario)
                inputs.push([(lat + 90) / 180, (long + 180) / 360, velocidad / 10]);

                if (row.Comportamiento !== undefined) {
                    const comportamiento = row.Comportamiento === 'Normal' ? 0 : 1;
                    labels.push(comportamiento); // Asegúrate de que labels sea un array plano
                }
            }
        });

        return {
            inputs: tf.tensor2d(inputs),
            labels: labels.length > 0 ? tf.tensor2d(labels, [labels.length, 1]) : null, // Especificar la forma de los labels
        };
    };

    const createAndTrainModel = async () => {
        const { inputs, labels } = await loadData('/comportamiento_ganado_simulado.csv');
        const model = await createModel();

        if (labels) {
            await trainModel(model, inputs, labels);
        }

        // Cargar los datos de prueba y hacer predicciones
        const { inputs: testInputs } = await loadData('/comportamiento_ganado_simulado_prueba.csv');
        makePredictions(model, testInputs);
    };

    const createModel = async () => {
        const model = tf.sequential();
        model.add(tf.layers.dense({ units: 16, activation: 'relu', inputShape: [3] }));
        model.add(tf.layers.dense({ units: 8, activation: 'relu' }));
        model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));

        model.compile({ optimizer: tf.train.adam(0.001), loss: 'binaryCrossentropy', metrics: ['accuracy'] });

        return model;
    };

    const trainModel = async (model, inputs, labels) => {
        await model.fit(inputs, labels, {
            epochs: 200,
            batchSize: 32,
            validationSplit: 0.2,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    console.log(`Epoch: ${epoch}, Loss: ${logs.loss.toFixed(4)}, Accuracy: ${logs.acc.toFixed(4)}`);
                },
            },
        });
    };

    const makePredictions = async (model, inputs) => {
        const prediccionesTensor = model.predict(inputs);
        const prediccionesArray = await prediccionesTensor.array();

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
