const latInput = document.querySelector(".lat-input"); // Input field for latitude
const lonInput = document.querySelector(".lon-input"); // Input field for longitude
const searchButton = document.querySelector(".search-btn");

const API_KEY = "4f2f9351bb4603c4dc21c0bd094a2bb3"; // API key

const getCityByCoordinates = async () => {
    const latitude = latInput.value.trim(); // Latitude
    const longitude = lonInput.value.trim(); // Longitude

    // Input validation for latitude and longitude
    if (!latitude || !longitude || isNaN(latitude) || isNaN(longitude)) {
        console.error("Please enter valid coordinates!");
        return;
    }

    const REVERSE_GEOCODING_API_URL = `https://api.openweathermap.org/geo/1.0/reverse?lat=${latitude}&lon=${longitude}&limit=1&appid=${API_KEY}`;

    try {
        // Fetching data from the API
        const response = await fetch(REVERSE_GEOCODING_API_URL);
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        const data = await response.json();
        
        // Log the raw response for debugging
        console.log("API Response:", data);

        // Checking if valid data is returned
        if (data.length > 0) {
            const { name, country } = data[0];
            // Logging the location in the console
            console.log(`Location: ${name}, ${country}`);
        } else {
            console.log("Location not found!");
        }
    } catch (error) {
        console.error("An error occurred while fetching the coordinates:", error);
    }
};

// Add event listener for search button
searchButton.addEventListener("click", getCityByCoordinates);
