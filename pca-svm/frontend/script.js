async function uploadFile() {
    const fileInput = document.getElementById('csvFile');
    
    if (!fileInput) { console.error("Input non trovato"); return; }
    const file = fileInput.files[0];
    if (!file) { alert("Per favore seleziona un file CSV."); return; }

    // UI Updates
    const loader = document.getElementById('loader');
    const resultArea = document.getElementById('resultArea');
    if (loader) loader.classList.remove('hidden');
    if (resultArea) resultArea.classList.add('hidden');

    const formData = new FormData();
    formData.append("file", file);

    try {
        // Percorso relativo (funziona grazie all'Ingress unificato)
        const response = await fetch("/upload/", { 
            method: "POST",
            body: formData
        });

        // 1. Controlla se la risposta HTTP è OK (200)
        if (!response.ok) {
            throw new Error(`Errore HTTP Server: ${response.status} ${response.statusText}`);
        }

        const data = await response.json();
        
        // --- DEBUG: GUARDA QUI NELLA CONSOLE ---
        console.log("RISPOSTA DAL SERVER:", data);

        if (data.error) {
            alert("Errore dal Backend: " + data.error);
        } else {
            // 2. RENDIAMO IL CODICE ROBUSTO
            // Se c'è data.result usa quello, altrimenti usa data intero
            const finalResult = data.result || data;
            
            // 3. Controllo finale prima di passare i dati
            if (finalResult && finalResult.accuracy !== undefined) {
                showResults(finalResult);
            } else {
                console.error("Struttura JSON imprevista:", data);
                alert("Analisi completata, ma il formato dei dati è imprevisto. Apri la Console (F12) per i dettagli.");
            }
        }
    } catch (error) {
        alert("Si è verificato un errore: " + error.message);
        console.error(error);
    } finally {
        if (loader) loader.classList.add('hidden');
    }
}

function showResults(res) {
    const resultArea = document.getElementById('resultArea');
    if (resultArea) resultArea.classList.remove('hidden');

    // Helper per aggiornare il testo in sicurezza
    const setTx = (id, txt) => {
        const el = document.getElementById(id);
        if (el) el.textContent = txt;
    };

    setTx('accuracy', (res.accuracy * 100).toFixed(2) + "%");
    setTx('f1', res.f1_macro.toFixed(3));
    setTx('folds', res.n_folds);
    setTx('variance', JSON.stringify(res.explained_var_cumsum, null, 2));
    setTx('matrix', JSON.stringify(res.confusion_matrix, null, 2));
}

// --- EVENT LISTENER PROTETTO ---
// Questo codice parte SOLO quando la pagina è pronta
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM caricato completamente");

    const fileInput = document.getElementById('csvFile');
    if (fileInput) {
        fileInput.addEventListener('change', function(e) {
            const fileName = e.target.files[0] ? e.target.files[0].name : "Nessun file selezionato";
            const nameLabel = document.getElementById('fileName');
            if (nameLabel) nameLabel.textContent = fileName;
        });
    } else {
        console.error("ERRORE CRITICO: Input file 'csvFile' non trovato nell'HTML!");
    }
});
