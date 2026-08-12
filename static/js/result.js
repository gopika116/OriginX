document.addEventListener("DOMContentLoaded", function () {

    console.log("=================================");
    console.log("OriginX AI Result Page Loaded");
    console.log("=================================");


    // =========================================
    // DOWNLOAD REPORT BUTTON
    // =========================================

    const downloadButton =
        document.querySelector(".btn-download");


    if (downloadButton) {

        downloadButton.addEventListener(
            "click",
            function () {

                window.print();

            }
        );

    }


    // =========================================
    // RESULT PAGE ANIMATION
    // =========================================

    const cards =
        document.querySelectorAll(".glass-card");


    cards.forEach(function (card, index) {

        card.style.opacity = "0";

        card.style.transform =
            "translateY(20px)";


        setTimeout(function () {

            card.style.transition =
                "all 0.6s ease";

            card.style.opacity = "1";

            card.style.transform =
                "translateY(0)";

        }, index * 120);

    });


    // =========================================
    // TIMELINE ANIMATION
    // =========================================

    const steps =
        document.querySelectorAll(".step");


    steps.forEach(function (step, index) {

        setTimeout(function () {

            step.style.transition =
                "all 0.5s ease";

            step.style.opacity = "1";

        }, index * 200);

    });


    // =========================================
    // CONSOLE MESSAGE
    // =========================================

    console.log(
        "OriginX AI: Result dashboard ready."
    );

});