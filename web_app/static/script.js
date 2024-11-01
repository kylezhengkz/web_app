async function submitData() {
    try {
        const inputData = document.getElementById("inputData").value;
        const resultElement = document.getElementById("result");
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ input: inputData })
        });
        const result = await response.json();
        resultElement.innerText = "Prediction: ${result.result}";
    } catch (error) {
        resultElement.innerText = "Error: Could not get prediction.";
        console.error("Error:", error);
    }
}
