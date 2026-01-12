// theme-toggle.js
document.addEventListener("DOMContentLoaded", function () {
  const body = document.body;
  const sunIcon = document.getElementById("sun");
  const moonIcon = document.getElementById("moon");
  let preferredTheme = localStorage.getItem("theme");

  // Apply the preferred theme or default to dark mode
  if (preferredTheme === "light") {
    body.classList.add("light");
    sunIcon.style.display = "block"; // Show sun icon for light mode
    moonIcon.style.display = "none"; // Hide moon icon
  } else {
    body.classList.add("dark");
    sunIcon.style.display = "none"; // Hide sun icon
    moonIcon.style.display = "block"; // Show moon icon for dark mode
  }

  document.getElementById("theme-toggle").addEventListener("click", () => {
    console.log("toggling");
    body.classList.toggle("light");
    body.classList.toggle("dark");

    // Update the preference in localStorage
    const newTheme = body.classList.contains("light") ? "light" : "dark";
    if (newTheme === "light") {
      sunIcon.style.display = "block"; // Show sun icon for light mode
      moonIcon.style.display = "none"; // Hide moon icon
    } else {
      sunIcon.style.display = "none"; // Hide sun icon
      moonIcon.style.display = "block"; // Show moon icon for dark mode
    }
    console.log("new theme:", newTheme);
    localStorage.setItem("theme", newTheme);
  });
});
