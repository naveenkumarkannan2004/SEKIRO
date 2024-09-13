// Example: Toggle navigation menu on small screens
document.addEventListener('DOMContentLoaded', function () {
    const navToggle = document.querySelector('.nav-toggle');
    const navMenu = document.querySelector('nav');

    if (navToggle) {
        navToggle.addEventListener('click', function () {
            navMenu.classList.toggle('hidden');
        });
    }
});

// Example: Display alert details on click
const alertBoxes = document.querySelectorAll('.alert-box');

alertBoxes.forEach((alert) => {
    alert.addEventListener('click', () => {
        alert.classList.toggle('bg-red-200'); // Toggle background color on click
    });
});
