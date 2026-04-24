import React from 'react';

const HospitalCard = ({ facility, address, town, county, type, the_geom }) => {
    const getDirections = () => {
        // Parse the_geom: POINT(lng lat)
        const match = the_geom.match(/POINT\(([^ ]+) ([^)]+)\)/);
        if (match) {
            const lng = match[1];
            const lat = match[2];
            const url = `https://www.google.com/maps/search/?api=1&query=${lat},${lng}`;
            window.open(url, '_blank');
        }
    };
    return (
        <div className="bg-white rounded-lg shadow-md p-4 border border-slate-200 hover:shadow-lg transition-shadow">
            <div className="flex items-start justify-between mb-2">
                <h3 className="font-bold text-lg text-slate-800">{facility}</h3>
                <span className={`px-2 py-1 text-xs font-semibold rounded-full ${
                    type === 'regional' ? 'bg-blue-100 text-blue-800' : 'bg-green-100 text-green-800'
                }`}>
                    {type}
                </span>
            </div>
            <p className="text-slate-600 text-sm mb-1">{address}</p>
            <p className="text-slate-500 text-sm mb-2">{town}, {county}</p>
            <div className="flex items-center justify-between">
                <span className="text-xs text-slate-400">Walk-in Available</span>
                <button 
                    onClick={getDirections}
                    className="bg-blue-600 text-white px-3 py-1 rounded text-sm hover:bg-blue-700 transition-colors"
                >
                    Get Directions
                </button>
            </div>
        </div>
    );
};

export default HospitalCard;