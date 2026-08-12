document.addEventListener("DOMContentLoaded", () => {

    const fileInput = document.getElementById("file");
    const uploadBox = document.getElementById("uploadBox");
    const fileName = document.getElementById("fileName");
    const preview = document.getElementById("preview");
    const analyzeBtn = document.getElementById("analyzeBtn");
    const mediaType = document.getElementById("mediaType");
    const previewContainer = document.getElementById("previewContainer");

    // ==========================
    // File Selection
    // ==========================
    fileInput.addEventListener("change", () => {
        handleFile(fileInput.files[0]);
    });

    // ==========================
    // Drag & Drop
    // ==========================
    uploadBox.addEventListener("dragover", (e) => {
        e.preventDefault();
        uploadBox.classList.add("dragging");
    });

    uploadBox.addEventListener("dragleave", () => {
        uploadBox.classList.remove("dragging");
    });

    uploadBox.addEventListener("drop", (e) => {
        e.preventDefault();
        uploadBox.classList.remove("dragging");

        const file = e.dataTransfer.files[0];

        if (file) {
            fileInput.files = e.dataTransfer.files;
            handleFile(file);
        }
    });

    // ==========================
    // Handle Selected File
    // ==========================
    function handleFile(file) {

        if (!file) return;

        fileName.textContent = file.name;

        const type = mediaType.value;

        if (type === "image") {

            previewContainer.style.display = "block";

            const reader = new FileReader();

            reader.onload = function(e) {
                preview.src = e.target.result;
            }

            reader.readAsDataURL(file);

        } else {

            previewContainer.style.display = "none";

        }

    }

    // ==========================
    // Analyze Button Animation
    // ==========================
    analyzeBtn.addEventListener("click", function () {

        if (fileInput.files.length === 0) {

            alert("Please choose a file first!");

            return;

        }

        analyzeBtn.disabled = true;

        analyzeBtn.innerHTML =
            '<span class="spinner"></span> Analyzing...';

    });

    // ==========================
    // Change Upload Text
    // ==========================
    mediaType.addEventListener("change", () => {

        const uploadText = document.getElementById("uploadText");

        switch(mediaType.value){

            case "image":
                uploadText.innerHTML =
                "Drag & Drop Image Here";
                break;

            case "video":
                uploadText.innerHTML =
                "Drag & Drop Video Here";
                break;

            case "audio":
                uploadText.innerHTML =
                "Drag & Drop Audio Here";
                break;

        }

        previewContainer.style.display = "none";
        fileName.textContent = "No file selected";

    });

    // ==========================
    // Navbar Shadow
    // ==========================
    window.addEventListener("scroll", () => {

        const nav = document.querySelector("nav");

        if (!nav) return;

        if(window.scrollY > 50){

            nav.classList.add("nav-scroll");

        }else{

            nav.classList.remove("nav-scroll");

        }

    });

    // ==========================
    // Animated Counters
    // ==========================
    const counters = document.querySelectorAll(".counter");

    counters.forEach(counter => {

        const target = +counter.dataset.target;

        let count = 0;

        const speed = target / 100;

        function update(){

            if(count < target){

                count += speed;

                counter.innerText = Math.floor(count);

                requestAnimationFrame(update);

            }else{

                counter.innerText = target;

            }

        }

        update();

    });

    // ==========================
    // Fade-in Animation
    // ==========================
    const reveals = document.querySelectorAll(".reveal");

    function revealSections(){

        reveals.forEach(section=>{

            const top = section.getBoundingClientRect().top;

            const windowHeight = window.innerHeight;

            if(top < windowHeight - 100){

                section.classList.add("active");

            }

        });

    }

    revealSections();

    window.addEventListener("scroll", revealSections);

});