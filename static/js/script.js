const form = document.getElementById("predictionForm");

if (form) {
  form.addEventListener("submit", (event) => {
    const area = Number(document.getElementById("area_sqft").value);
    if (!Number.isFinite(area) || area <= 0) {
      event.preventDefault();
      alert("Please enter a valid area greater than zero.");
    }
  });
}
