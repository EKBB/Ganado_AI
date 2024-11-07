import React, { useState, useEffect } from 'react';
import * as tf from '@tensorflow/tfjs';

function CoordenadasApp() {
    const [latitud, setLatitud] = useState('');
    const [longitud, setLongitud] = useState('');
    const [coordenadas, setCoordenadas] = useState([]);
    const [distanciaTotal, setDistanciaTotal] = useState(0);
    const [caloriasGastadas, setCaloriasGastadas] = useState(0);
    const [prediccion, setPrediccion] = useState(null);
    const [recomendacion, setRecomendacion] = useState('');
    const [modelo, setModelo] = useState(null);
    const [coordenadasRepetidas, setCoordenadasRepetidas] = useState({});

    useEffect(() => {
        const entrenarModelo = async () => {
            // Datos de entrenamiento con coordenadas y etiquetas alineadas
            const datosEntrenamiento = tf.tensor2d([
                [1, 1], [1.1, 1.1], [1.2, 1.2], [10, 10], [11, 11], [12, 12], 
                [13, 13], [1.5, 1.5], [0.5, 0.5], [20, 20], [15, 15], [5, 5],
                [5.5, 5.5], [6, 6]
            ]);
            const etiquetas = tf.tensor2d([[0], [0], [0], [1], [1], [1], [1], [0], [0], [1], [1], [0], [1], [0]]); // Asegúrate de que las etiquetas coincidan

            const model = tf.sequential();
            model.add(tf.layers.dense({ inputShape: [2], units: 64, activation: 'relu' }));
            model.add(tf.layers.dense({ units: 32, activation: 'relu' }));
            model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' }));
            model.compile({
                optimizer: tf.train.adam(0.001),
                loss: 'binaryCrossentropy',
                metrics: ['accuracy']
            });

            await model.fit(datosEntrenamiento, etiquetas, {
                epochs: 200,
                batchSize: 32,
                callbacks: {
                    onEpochEnd: (epoch, logs) => console.log(`Epoch ${epoch + 1}: Loss = ${logs.loss}`)
                }
            });

            setModelo(model);
        };

        entrenarModelo();
    }, []);

    const agregarCoordenada = async () => {
        if (latitud && longitud) {
            const nuevaCoordenada = { latitud: parseFloat(latitud), longitud: parseFloat(longitud) };

            // Convertir la coordenada a una clave única
            const key = `${nuevaCoordenada.latitud},${nuevaCoordenada.longitud}`;

            // Actualizar el contador de coordenadas repetidas
            setCoordenadasRepetidas(prev => {
                const newCounts = { ...prev };
                newCounts[key] = (newCounts[key] || 0) + 1;

                // Si la coordenada se repite 3 veces, mostrar una alerta
                if (newCounts[key] === 3) {
                    alert("¡El animal ha estado en la misma ubicación durante demasiado tiempo!");
                }

                return newCounts;
            });

            // Calcular distancia y calorías si ya hay coordenadas
            if (coordenadas.length > 0) {
                const distancia = calcularDistancia(
                    coordenadas[coordenadas.length - 1],
                    nuevaCoordenada
                );
                setDistanciaTotal(distanciaTotal + distancia);
                
                const calorias = calcularCalorias(distancia);
                setCaloriasGastadas(caloriasGastadas + calorias);

                generarRecomendacion(caloriasGastadas + calorias);
            }

            // Agregar nueva coordenada
            setCoordenadas([...coordenadas, nuevaCoordenada]);

            // Actualizar predicción si hay un modelo entrenado
            if (modelo) {
                const input = tf.tensor2d([[nuevaCoordenada.latitud, nuevaCoordenada.longitud]]);
                const resultado = await modelo.predict(input).array();
                setPrediccion(resultado[0][0] > 0.5 ? "Desviado" : "Dentro del grupo");
            }

            setLatitud('');
            setLongitud('');
        } else {
            alert("Por favor ingresa valores para latitud y longitud.");
        }
    };

    const calcularDistancia = (coord1, coord2) => {
        const R = 6371; // Radio de la tierra en km
        const dLat = (coord2.latitud - coord1.latitud) * (Math.PI / 180);
        const dLon = (coord2.longitud - coord1.longitud) * (Math.PI / 180);
        const a =
            Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(coord1.latitud * (Math.PI / 180)) *
            Math.cos(coord2.latitud * (Math.PI / 180)) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2);
        const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
        return R * c; // Distancia en km
    };

    const calcularCalorias = (distancia) => {
        const pesoPromedio = 50; // Peso del animal en kg (ajustado para pruebas)
        const factorCalorico = 0.05; // Calorías por km por kg de peso (ajustado para pruebas)
        return distancia * pesoPromedio * factorCalorico;
    };

    const generarRecomendacion = (calorias) => {
        if (calorias < 500) {
            setRecomendacion("El animal puede continuar su actividad.");
        } else if (calorias >= 500 && calorias < 1000) {
            setRecomendacion("Recomendar descanso al animal.");
        } else {
            setRecomendacion("Regresar al animal al grupo para evitar agotamiento.");
        }
    };

   

    return (
        <div style={{ padding: '20px' }}>
            <h2>Ingresar Coordenadas</h2>
            <div>
                <label>Latitud: </label>
                <input
                    type="text"
                    value={latitud}
                    onChange={(e) => setLatitud(e.target.value)}
                />
            </div>
            <div>
                <label>Longitud: </label>
                <input
                    type="text"
                    value={longitud}
                    onChange={(e) => setLongitud(e.target.value)}
                />
            </div>
            <button onClick={agregarCoordenada} style={{ marginTop: '10px' }}>
                Agregar Coordenada y Clasificar
            </button>

            <h3>Lista de Coordenadas</h3>
            <ul>
                {coordenadas.map((coord, index) => (
                    <li key={index}>
                        Latitud: {coord.latitud}, Longitud: {coord.longitud}
                    </li>
                ))}
            </ul>

            {prediccion && (
                <div>
                    <h3>Predicción</h3>
                    <p>El animal está: <strong>{prediccion}</strong></p>
                </div>
            )}

            <h3>Distancia Total Recorrida</h3>
            <p>{distanciaTotal.toFixed(2)} km</p>

            <h3>Calorías Gastadas</h3>
            <p>{caloriasGastadas.toFixed(2)} calorías</p>

            {recomendacion && (
                <div>
                    <h3>Recomendación</h3>
                    <p>{recomendacion}</p>
                </div>
            )}
        </div>
    );
}

export default CoordenadasApp;
