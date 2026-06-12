/* =====================================================
   DOM ELEMENTS
===================================================== */

const dropArea = document.getElementById("dropArea");
const fileInput = document.getElementById("fileInput");
const browseBtn = document.getElementById("browseBtn");
const removeImageBtn = document.getElementById("removeImage");
const previewContainer = document.getElementById("previewContainer");
const previewImage = document.getElementById("previewImage");
const analyzeBtn = document.getElementById("analyzeBtn");

const loadingSection = document.getElementById("loadingSection");
const progressBar = document.getElementById("progressBar");

const resultsSection = document.getElementById("results");
const comparisonSection = document.getElementById("comparisonSection");

const predictionText = document.getElementById("predictionText");
const confidenceValue = document.getElementById("confidenceValue");
const confidenceBar = document.getElementById("confidenceBar");
const predictionCard = document.getElementById("predictionCard");

const originalImage = document.getElementById("originalImage");
const heatmapImage = document.getElementById("heatmapImage");

const downloadOriginal = document.getElementById("downloadOriginal");
const downloadHeatmap = document.getElementById("downloadHeatmap");

const toastContainer = document.getElementById("toastContainer");

const modal = document.getElementById("imageModal");
const modalImage = document.getElementById("modalImage");
const closeModal = document.getElementById("closeModal");

const hamburger = document.getElementById("hamburger");
const navLinks = document.getElementById("navLinks");

/* =====================================================
   STATE
===================================================== */

let selectedFile = null;

/* =====================================================
   MOBILE MENU
===================================================== */

hamburger?.addEventListener("click", () => {
    navLinks.classList.toggle("active");
});

/* =====================================================
   BROWSE BUTTON
===================================================== */

browseBtn.addEventListener("click", (e) => {
    e.preventDefault();
    fileInput.click();
});

/* =====================================================
   FILE INPUT
===================================================== */

fileInput.addEventListener("change", (e) => {
    const file = e.target.files[0];

    if (file) {
        validateAndPreview(file);
    }
});

/* =====================================================
   DRAG & DROP
===================================================== */

["dragenter", "dragover"].forEach(eventName => {
    dropArea.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropArea.classList.add("dragover");
    });
});

["dragleave", "drop"].forEach(eventName => {
    dropArea.addEventListener(eventName, (e) => {
        e.preventDefault();
        dropArea.classList.remove("dragover");
    });
});

dropArea.addEventListener("drop", (e) => {

    const file = e.dataTransfer.files[0];

    if (file) {
        validateAndPreview(file);
    }
});

/* =====================================================
   VALIDATION
===================================================== */

function validateAndPreview(file) {

    const validTypes = [
        "image/jpeg",
        "image/jpg",
        "image/png"
    ];

    const maxSize = 10 * 1024 * 1024;

    if (!validTypes.includes(file.type)) {

        showToast(
            "Only JPG, JPEG, PNG files allowed",
            "error"
        );

        return;
    }

    if (file.size > maxSize) {

        showToast(
            "File exceeds 10MB limit",
            "error"
        );

        return;
    }

    selectedFile = file;

    const reader = new FileReader();

    reader.onload = function (e) {

        previewImage.src = e.target.result;

        previewContainer.classList.remove("hidden");

        showToast(
            "Image loaded successfully",
            "success"
        );
    };

    reader.readAsDataURL(file);
}

/* =====================================================
   REMOVE IMAGE
===================================================== */

removeImageBtn.addEventListener("click", () => {

    selectedFile = null;

    fileInput.value = "";

    previewImage.src = "";

    previewContainer.classList.add("hidden");

    showToast(
        "Image removed",
        "info"
    );
});

/* =====================================================
   ANALYZE BUTTON
===================================================== */

analyzeBtn.addEventListener("click", async () => {

    if (!selectedFile) {

        showToast(
            "Please upload an X-Ray image",
            "error"
        );

        return;
    }

    try {

        startLoading();

        const formData = new FormData();

        formData.append(
            "file",
            selectedFile
        );

        const response = await fetch(
            "/predict",
            {
                method: "POST",
                body: formData
            }
        );

        if (!response.ok) {
            throw new Error("Server Error");
        }

        const data = await response.json();

        renderResults(data);

        stopLoading();

        showToast(
            "Analysis completed successfully",
            "success"
        );

    } catch (error) {

        console.error(error);

        stopLoading();

        showToast(
            "Prediction failed. Please try again.",
            "error"
        );
    }
});

/* =====================================================
   LOADING
===================================================== */

function startLoading() {

    resultsSection.classList.add("hidden");
    comparisonSection.classList.add("hidden");

    loadingSection.classList.remove("hidden");

    analyzeBtn.disabled = true;

    let width = 0;

    const interval = setInterval(() => {

        width += 5;

        progressBar.style.width =
            width + "%";

        if (width >= 95) {
            clearInterval(interval);
        }

    }, 250);

    window.loadingInterval = interval;
}

function stopLoading() {

    clearInterval(
        window.loadingInterval
    );

    progressBar.style.width = "100%";

    setTimeout(() => {

        loadingSection.classList.add(
            "hidden"
        );

        analyzeBtn.disabled = false;

        progressBar.style.width = "0%";

    }, 800);
}

/* =====================================================
   RENDER RESULTS
===================================================== */

function renderResults(data) {

    resultsSection.classList.remove(
        "hidden"
    );

    comparisonSection.classList.remove(
        "hidden"
    );

    predictionText.textContent =
        data.prediction;

    predictionText.classList.remove(
        "prediction-normal",
        "prediction-pneumonia"
    );

    predictionCard.classList.remove(
        "success-glow",
        "danger-glow"
    );

    if (
        data.prediction
            .toLowerCase()
            .includes("pneumonia")
    ) {

        predictionText.classList.add(
            "prediction-pneumonia"
        );

        predictionCard.classList.add(
            "danger-glow"
        );

    } else {

        predictionText.classList.add(
            "prediction-normal"
        );

        predictionCard.classList.add(
            "success-glow"
        );
    }

    animateConfidence(
        Number(data.confidence)
    );

    originalImage.src =
        data.original_image;

    heatmapImage.src =
        data.heatmap_image;

    attachModal(originalImage);
    attachModal(heatmapImage);

    setupDownloads(
        data.original_image,
        data.heatmap_image
    );

    resultsSection.scrollIntoView({
        behavior: "smooth"
    });
}

/* =====================================================
   CONFIDENCE ANIMATION
===================================================== */

function animateConfidence(value) {

    let start = 0;

    const duration = 1500;

    const stepTime = 15;

    const increment =
        value /
        (duration / stepTime);

    confidenceBar.style.width =
        value + "%";

    const interval = setInterval(() => {

        start += increment;

        if (start >= value) {

            start = value;

            clearInterval(interval);
        }

        confidenceValue.textContent =
            start.toFixed(1) + "%";

    }, stepTime);
}

/* =====================================================
   DOWNLOADS
===================================================== */

function setupDownloads(
    originalUrl,
    heatmapUrl
) {

    downloadOriginal.onclick = () => {

        downloadFile(
            originalUrl,
            "original_xray.jpg"
        );
    };

    downloadHeatmap.onclick = () => {

        downloadFile(
            heatmapUrl,
            "gradcam_heatmap.jpg"
        );
    };
}

function downloadFile(url, filename) {

    const link =
        document.createElement("a");

    link.href = url;

    link.download = filename;

    document.body.appendChild(link);

    link.click();

    document.body.removeChild(link);
}

/* =====================================================
   MODAL
===================================================== */

function attachModal(imageElement) {

    imageElement.onclick = () => {

        modal.classList.add("show");

        modalImage.src =
            imageElement.src;
    };
}

closeModal.addEventListener(
    "click",
    () => {

        modal.classList.remove("show");
    }
);

modal.addEventListener(
    "click",
    (e) => {

        if (e.target === modal) {

            modal.classList.remove(
                "show"
            );
        }
    }
);

/* =====================================================
   TOASTS
===================================================== */

function showToast(
    message,
    type = "info"
) {

    const toast =
        document.createElement("div");

    toast.className =
        `toast toast-${type}`;

    let icon = "fa-circle-info";

    if (type === "success")
        icon = "fa-circle-check";

    if (type === "error")
        icon = "fa-circle-xmark";

    toast.innerHTML = `
        <i class="fa-solid ${icon}"></i>
        <span>${message}</span>
    `;

    toastContainer.appendChild(
        toast
    );

    setTimeout(() => {

        toast.style.opacity = "0";

        toast.style.transform =
            "translateX(100%)";

        setTimeout(() => {

            toast.remove();

        }, 400);

    }, 3500);
}

/* =====================================================
   ACTIVE NAV LINK
===================================================== */

const sections =
    document.querySelectorAll("section");

const navItems =
    document.querySelectorAll(".nav-link");

window.addEventListener(
    "scroll",
    () => {

        let current = "";

        sections.forEach(section => {

            const top =
                section.offsetTop - 150;

            if (
                pageYOffset >= top
            ) {
                current =
                    section.getAttribute(
                        "id"
                    );
            }
        });

        navItems.forEach(link => {

            link.classList.remove(
                "active"
            );

            if (
                link.getAttribute("href")
                === "#" + current
            ) {

                link.classList.add(
                    "active"
                );
            }
        });
    }
);

/* =====================================================
   SCROLL ANIMATION
===================================================== */

const observer =
    new IntersectionObserver(
        entries => {

            entries.forEach(entry => {

                if (
                    entry.isIntersecting
                ) {

                    entry.target.classList.add(
                        "show"
                    );
                }
            });

        },
        {
            threshold: 0.15
        }
    );

document
.querySelectorAll(
    ".section-header,.glass,.result-card"
)
.forEach(el => {

    el.classList.add(
        "fade-in"
    );

    observer.observe(el);
});

/* =====================================================
   INIT
===================================================== */

showToast(
    "AI Pneumonia Detection System Ready",
    "info"
);


