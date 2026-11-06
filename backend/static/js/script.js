
document.addEventListener("DOMContentLoaded", () => {

    // ==========================================
    // GET HTML ELEMENTS
    // ==========================================

    const imageInput = document.getElementById("image-input");
    const chooseButton = document.getElementById("choose-button");
    const uploadArea = document.getElementById("upload-area");

    const previewContainer = document.getElementById("preview-container");
    const imagePreview = document.getElementById("image-preview");
    const removeButton = document.getElementById("remove-button");

    const verifyButton = document.getElementById("verify-button");

    const loading = document.getElementById("loading");
    const result = document.getElementById("result");
    const errorMessage = document.getElementById("error-message");

    // Prediction fields
    const prediction = document.getElementById("prediction");
    const realProbability = document.getElementById("real-probability");
    const fakeProbability = document.getElementById("fake-probability");
    const modelConfidence = document.getElementById("model-confidence");

    const verificationId = document.getElementById("verification-id");
    const resultNote = document.getElementById("result-note");

    let selectedFile = null;
    let previewURL = null;


    // ==========================================
    // FORMAT PERCENTAGE
    // ==========================================

    function formatPercentage(value) {

        const number = Number(value);

        if (!Number.isFinite(number)) {
            return "--";
        }

        return number.toFixed(2) + "%";
    }


    // ==========================================
    // CHOOSE IMAGE
    // ==========================================

    chooseButton.addEventListener("click", () => {

        imageInput.click();

    });


    // ==========================================
    // IMAGE SELECTED
    // ==========================================

    imageInput.addEventListener("change", () => {

        const file = imageInput.files[0];

        if (!file) {
            return;
        }

        if (!["image/jpeg", "image/png"].includes(file.type)) {

            alert("Please select a JPG or PNG image.");

            imageInput.value = "";

            return;
        }

        selectedFile = file;

        if (previewURL) {

            URL.revokeObjectURL(previewURL);

        }

        previewURL = URL.createObjectURL(file);

        imagePreview.src = previewURL;

        previewContainer.classList.remove("hidden");

        uploadArea.classList.add("hidden");

        verifyButton.disabled = false;

        result.classList.add("hidden");

        errorMessage.classList.add("hidden");

    });


    // ==========================================
    // REMOVE IMAGE
    // ==========================================

    removeButton.addEventListener("click", () => {

        selectedFile = null;

        imageInput.value = "";

        imagePreview.src = "";

        if (previewURL) {

            URL.revokeObjectURL(previewURL);

            previewURL = null;

        }

        previewContainer.classList.add("hidden");

        uploadArea.classList.remove("hidden");

        verifyButton.disabled = true;

        result.classList.add("hidden");

        errorMessage.classList.add("hidden");

    });


    // ==========================================
    // VERIFY IMAGE
    // ==========================================

    verifyButton.addEventListener("click", async () => {

        if (!selectedFile) {

            alert("Please select an image first.");

            return;

        }

        // Show loading
        loading.classList.remove("hidden");

        result.classList.add("hidden");

        errorMessage.classList.add("hidden");

        verifyButton.disabled = true;


        try {

            // Create FormData
            const formData = new FormData();

            formData.append("image", selectedFile);


            // Send image to Flask API
            const response = await fetch("/api/predict", {

                method: "POST",

                body: formData

            });


            // Read JSON response
            const data = await response.json();


            // Check response
            if (!response.ok || !data.success) {

                throw new Error(
                    data.message || "Prediction failed."
                );

            }


            // ==========================================
            // DISPLAY PREDICTION
            // ==========================================

            prediction.textContent =
                data.prediction || "--";


            // Display real probability
            realProbability.textContent =
                formatPercentage(data.real_probability);


            // Display fake probability
            fakeProbability.textContent =
                formatPercentage(data.fake_probability);


            // Display model confidence
            modelConfidence.textContent =
                formatPercentage(data.confidence);


            // ==========================================
            // GENERATE VERIFICATION ID
            // ==========================================

            const verificationID =
                "VF-" +
                Date.now().toString(36).toUpperCase();

            verificationId.textContent =
                verificationID;


            // ==========================================
            // RESULT MESSAGE
            // ==========================================

            resultNote.textContent =
                data.message ||
                "Image analysis completed using the VeraFace EfficientNet-B0 model.";


            // Show result
            result.classList.remove("hidden");


        } catch (error) {

            console.error("Prediction error:", error);

            errorMessage.textContent =
                error.message ||
                "Unable to verify the image.";

            errorMessage.classList.remove("hidden");


        } finally {

            // Hide loading
            loading.classList.add("hidden");

            // Enable button if an image is selected
            verifyButton.disabled = !selectedFile;

        }

    });

});
