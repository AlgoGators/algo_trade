// This function could be connected to a real-time data feed from a broker API
function updatePositions() {
    const positions = [
        { instrument: 'CAD CME', last: 0.73655, change: -0.00050, position: -1, pl: -26.13 },
        { instrument: 'M6A CME', last: 0.68110, change: 0.00170, position: 1, pl: -0.71 },
        { instrument: 'MES CME', last: 5769.50, change: 89.50, position: 2, pl: 43.16 },
        { instrument: 'MGC CME', last: 2583.10, change: 8.20, position: 4, pl: -20.08 },
        { instrument: 'MNQ CME', last: 20091.50, change: 511.00, position: 1, pl: 57.88 },
        { instrument: 'ZR CBOT', last: 15.460, change: -0.005, position: 1, pl: -12.97 },
    ];

    let tableBody = document.querySelector('tbody');

    positions.forEach(position => {
        let row = `<tr class="text-center border-t">
                    <td class="py-2">${position.instrument}</td>
                    <td class="py-2">${position.last.toFixed(5)}</td>
                    <td class="py-2 ${position.change > 0 ? 'text-green-500' : 'text-red-500'}">${position.change.toFixed(5)}</td>
                    <td class="py-2">${position.position}</td>
                    <td class="py-2 ${position.pl > 0 ? 'text-green-500' : 'text-red-500'}">${position.pl.toFixed(2)}</td>
                </tr>`;
        tableBody.innerHTML += row;
    });
}

// Call the function to update the positions
window.onload = updatePositions;
