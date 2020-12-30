var slideIndex = 1;

showSlides(slideIndex);

function plusSlides(n) {
    showSlides(slideIndex += n);
}

function currentSlide(n) {
    showSlides(slideIndex = n);
}

function showSlides(n) {
    var i;
    var slides = document.getElementsByClassName("mySlides");

    if (n > slides.length) {slideIndex = 1}
    if (n < 1) {slideIndex = slides.length}
    for (i = 0; i < slides.length; i++) {
        slides[i].style.display = "none";
    }

    slides[slideIndex-1].style.display = "block";

}

var slideIndexB= 1
showSlidesB(slideIndexB)

function plusSlidesB(n) {
    showSlidesB(slideIndexB += n);
}

function currentSlideB(n) {
    showSlidesB(slideIndexB = n);
}

function showSlidesB(n) {
    var i;
    var slides = document.getElementsByClassName("mySlidesB");

    if (n > slides.length) {slideIndexB = 1}
    if (n < 1) {slideIndexB = slides.length}
    for (i = 0; i < slides.length; i++) {
        slides[i].style.display = "none";
    }

    slides[slideIndexB-1].style.display = "block";

}
