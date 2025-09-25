document.getElementById("upload-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const fileInput = document.getElementById("csv-file");
  const file = fileInput.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  const resultElement = document.getElementById("result");
  resultElement.textContent = "Uploading and processing...";

  try {
    const response = await fetch("/upload/", {
      method: "POST",
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`Errore ${response.status}: ${await response.text()}`);
    }

    const data = await response.json();
    resultElement.textContent = JSON.stringify(data, null, 2);
  } catch (err) {
    resultElement.textContent = `Errore: ${err.message}`;
  }
});
