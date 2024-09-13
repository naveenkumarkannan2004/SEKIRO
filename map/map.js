let map;

function initMap() {
    map = new google.maps.Map(document.getElementById("map"), {
        center: { lat: 13.08, lng: 80.27 },
        zoom: 15,
        mapTypeId: "satellite"
    });

    // Array of locations
    const locations = [
        { lat: 13.08, lng: 80.27, title: "Central",icon:"girl.png",animation:google.maps.Animation.DROP },
        { lat: 13.07, lng: 80.27, title: "Chindatripet", icon:"danger.png",animation:google.maps.Animation.BOUNCE },
        { lat: 13.11, lng: 80.29, title: "Royapuram",icon:"danger.png" ,animation:google.maps.Animation.BOUNCE},
        { lat: 13.09, lng: 80.26, title: "pulianthope" ,icon:"danger.png",animation:google.maps.Animation.BOUNCE},
        { lat: 13.09, lng: 80.25, title: "purasaiwalkam" ,icon:"danger.png",animation:google.maps.Animation.BOUNCE},
        { lat: 13.07, lng: 80.26, title: "Egmore" ,icon:"danger.png",animation:google.maps.Animation.BOUNCE},
        { lat: 13.0777, lng: 80.2614, title: "Egmore railwaystation" ,icon:"danger.png",animation:google.maps.Animation.BOUNCE},
        { lat: 13.0884, lng: 80.2838, title: "Flower bazar" ,icon:"danger.png",animation:google.maps.Animation.BOUNCE}
    ];

    // Loop through the array and create a marker for each location
    locations.forEach((location) => {
        const marker = new google.maps.Marker({
            position: { lat: location.lat, lng: location.lng },
            map: map,
            label: "",
            title: location.title,
            draggable: true,
            animation: location.animation,
            icon: location.icon // Replace with your custom icon path
        });

        // Optionally, you can add an info window for each marker
        const infoWindow = new google.maps.InfoWindow({
            content: `<h3>${location.title}</h3>`
        });

        marker.addListener("click", () => {
            infoWindow.open(map, marker);
        });
    });
}