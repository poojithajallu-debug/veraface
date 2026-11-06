const imageInput = document.getElementById("image-input");
const chooseButton = document.getElementById("choose-button");
const verifyButton = document.getElementById("verify-button");

const previewContainer = document.getElementById("preview-container");
const imagePreview = document.getElementById("image-preview");
const removeButton = document.getElementById("remove-button");

const result = document.getElementById("result");
const loading = document.getElementById("loading");

let selectedImage = null;


chooseButton.addEventListener("click", () => {
    imageInput.click();
});


imageInput.addEventListener("change", (event) => {

    const file = event.target.files[0];

    if (!file) {
        return;
    }

    selectedImage = file;

    const imageURL = URL.createObjectURL(file);

    imagePreview.src = imageURL;

    previewContainer.classList.remove("hidden");

    verifyButton.disabled = false;

    result.classList.add("hidden");
});


removeButton.addEventListener("click", () => {

    selectedImage = null;

    imageInput.value = "";

    imagePreview.src = "";

    previewContainer.classList.add("hidden");

    verifyButton.disabled = true;

    result.classList.add("hidden");
});


verifyButton.addEventListener("click", async () => {

    if (!selectedImage) {
        return;
    }

    loading.classList.remove("hidden");

    verifyButton.disabled = true;

    result.classList.add("hidden");

    /*
     * API connection will be added in the next step.
     */

    setTimeout(() => {

        loading.classList.add("hidden");

        verifyButton.disabled = false;

        result.classList.remove("hidden");

        document.getElementById("prediction").textContent = "Demo";
        document.getElementById("probability").textContent = "--";
        document.getElementById("trust-score").textContent = "--";
        document.getElementById("verification-id").textContent = "--";

        document.getElementById("result-note").textContent =
            "Frontend test completed. AI API connection will be added next.";

    }, 800);
});