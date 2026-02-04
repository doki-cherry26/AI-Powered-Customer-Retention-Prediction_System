function predict() {
    const payload = {
        tenure: document.getElementById("tenure").value,
        MonthlyCharges: document.getElementById("MonthlyCharges").value,
        TotalCharges: document.getElementById("TotalCharges").value,
        gender: document.getElementById("gender").value,
        SeniorCitizen: document.getElementById("SeniorCitizen").value,
        Partner: document.getElementById("Partner").value,
        Dependents: document.getElementById("Dependents").value,
        PhoneService: document.getElementById("PhoneService").value,
        PaperlessBilling: document.getElementById("PaperlessBilling").value,
        Contract: document.getElementById("Contract").value,
        PaymentMethod: document.getElementById("PaymentMethod").value,
        SimCard: document.getElementById("SimCard").value
    };

    fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    })
    .then(res => res.json())
    .then(data => {
        if (data.error) {
            document.getElementById("result").innerText = "Error: " + data.error;
        } else {
            document.getElementById("result").innerText =
                `Churn: ${data.churn} | Probability: ${data.probability}%`;
        }
    })
    .catch(err => {
        document.getElementById("result").innerText = "Server error";
        console.error(err);
    });
}
<script>
function selectSim(name) {
    document.getElementById("SimCard").value = name;

    document.querySelectorAll(".sim-card").forEach(card => {
        card.classList.remove("active");
    });
    static/images/airtel.png
static/images/jio.png
static/images/vi.png
static/images/bsnl.png


    event.currentTarget.classList.add("active");
}
</script>

SimCard: document.getElementById("SimCard").value



