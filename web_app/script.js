async function submitData() {
    const inputData = document.getElementById('inputData').value;
    const resultElement = document.getElementById('result');

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ input: inputData })
        });

        const result = await response.json();
        resultElement.innerText = `Prediction: ${result.result}`;
    } catch (error) {
        resultElement.innerText = 'Error: Could not get prediction.';
        console.error('Error:', error);
    }
}
