// import React, { useEffect } from 'react';
// import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
// import L from 'leaflet';
// import { MOCK_FACILITIES, NOVA_SCOTIA_CENTER, FacilityType } from '../constants';

// // // Fix for default Leaflet markers in React
// // delete L.Icon.Default.prototype._getIconUrl;
// // L.Icon.Default.mergeOptions({
// //     iconRetinaUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png',
// //     iconUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png',
// //     shadowUrl: 'https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png',
// // });

// // // Custom Icons based on Facility Type
// // const createCustomIcon = (color, iconClass) => {
// //     return L.divIcon({
// //         className: 'custom-leaflet-icon',
// //         html: `<div style="background-color: ${color}; width: 36px; height: 36px; border-radius: 50%; display: flex; align-items: center; justify-content: center; border: 3px solid white; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.2), 0 2px 4px -1px rgba(0,0,0,0.06);">
// //                     <i class="${iconClass}" style="color: white; font-size: 16px;"></i>
// //                 </div>`,
// //         iconSize: [36, 36],
// //         iconAnchor: [18, 18],
// //         popupAnchor: [0, -18]
// //     });
// // };

// // const icons = {
// //     [FacilityType.HOSPITAL]: createCustomIcon('#dc2626', 'fas fa-h-square'), // red-600
// //     [FacilityType.CLINIC]: createCustomIcon('#eab308', 'fas fa-user-doctor'), // yellow-500
// //     [FacilityType.PHARMACY]: createCustomIcon('#16a34a', 'fas fa-pills'), // green-600
// //     [FacilityType.UNKNOWN]: createCustomIcon('#64748b', 'fas fa-question'), // slate-500
// // };

// // // Custom User Icon
// // const userIcon = L.divIcon({
// //     className: 'custom-user-icon',
// //     html: `<div style="background-color: #3b82f6; width: 24px; height: 24px; border-radius: 50%; border: 3px solid white; box-shadow: 0 0 10px rgba(59, 130, 246, 0.8); position: relative;">
// //                 <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 8px; height: 8px; background-color: white; border-radius: 50%;"></div>
// //             </div>`,
// //     iconSize: [24, 24],
// //     iconAnchor: [12, 12],
// //     popupAnchor: [0, -12]
// // });

// // // Helper component to smoothly fly to recommended location or fit bounds if user location is available
// // const MapUpdater = ({ targetFacility, userLocation }) => {
// //     const map = useMap();

// //     useEffect(() => {
// //         if (targetFacility && userLocation) {
// //             const bounds = L.latLngBounds([
// //                 [userLocation.lat, userLocation.lng],
// //                 [targetFacility.lat, targetFacility.lng]
// //             ]);
// //             map.fitBounds(bounds, { padding: [50, 50], maxZoom: 14, animate: true, duration: 1.5 });
// //         } else if (targetFacility) {
// //             map.flyTo([targetFacility.lat, targetFacility.lng], 14, { duration: 1.5 });
// //         } else if (userLocation) {
// //             map.flyTo([userLocation.lat, userLocation.lng], 13, { duration: 1.5 });
// //         } else {
// //             map.flyTo(NOVA_SCOTIA_CENTER, 12, { duration: 1 });
// //         }
// //     }, [targetFacility, userLocation, map]);

// //     return null;
// // };

// // const MapView = ({ triageData, userLocation }) => {
// //     // Filter and find the best facility based on triage data
// //     const recommendedFacilityType = triageData?.facilityType;

// //     const relevantFacilities = triageData && recommendedFacilityType !== FacilityType.UNKNOWN
// //         ? MOCK_FACILITIES.filter(f => f.type === recommendedFacilityType)
// //         : MOCK_FACILITIES;

// //     // Simple logic to find the "best" facility (e.g., shortest wait time) within the filtered list
// //     const bestFacility = relevantFacilities.length > 0
// //         ? [...relevantFacilities].sort((a, b) => (a.waitTimeMins || 999) - (b.waitTimeMins || 999))[0]
// //         : null;

// //     // Format wait time
// //     const formatWaitTime = (mins) => {
// //         if (mins === undefined) return 'Unknown';
// //         if (mins < 60) return `${mins} mins`;
// //         const hours = Math.floor(mins / 60);
// //         const remainder = mins % 60;
// //         return `${hours}h ${remainder > 0 ? `${remainder}m` : ''}`;
// //     };

// //     return (
// //         <div className="h-full w-full relative bg-slate-200">
// //             <MapContainer
// //                 center={NOVA_SCOTIA_CENTER}
// //                 zoom={12}
// //                 style={{ height: '100%', width: '100%' }}
// //                 zoomControl={false}
// //             >
// //                 <TileLayer
// //                     attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
// //                     url="https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png"
// //                 />

// //                 <MapUpdater targetFacility={triageData && triageData.level > 0 ? bestFacility : null} userLocation={userLocation} />

// //                 {userLocation && (
// //                     <Marker position={[userLocation.lat, userLocation.lng]} icon={userIcon} zIndexOffset={1000}>
// //                         <Popup className="custom-popup">
// //                             <div className="p-2 text-center min-w-[120px]">
// //                                 <h3 className="font-bold text-slate-800 text-sm">Your Location</h3>
// //                                 <p className="text-[10px] text-slate-500 mt-1">Used for accurate routing</p>
// //                             </div>
// //                         </Popup>
// //                     </Marker>
// //                 )}

// //                 {MOCK_FACILITIES.map((facility) => {
// //                     const isRecommended = bestFacility?.id === facility.id && triageData?.level > 0;
// //                     const isDimmed = triageData?.level > 0 && recommendedFacilityType !== FacilityType.UNKNOWN && facility.type !== recommendedFacilityType;

// //                     if (isDimmed) return null; // Hide non-relevant facilities for clarity

// //                     return (
// //                         <Marker
// //                             key={facility.id}
// //                             position={[facility.lat, facility.lng]}
// //                             icon={icons[facility.type]}
// //                         >
// //                             <Popup className="custom-popup">
// //                                 <div className="p-1 min-w-[220px]">
// //                                     {isRecommended && (
// //                                         <div className="bg-green-100 text-green-800 text-xs font-bold px-2 py-1 rounded-md mb-2 inline-block">
// //                                             <i className="fas fa-check-circle mr-1"></i> Recommended
// //                                         </div>
// //                                     )}
// //                                     <h3 className="font-bold text-slate-800 text-base leading-tight mb-1">{facility.name}</h3>
// //                                     <p className="text-xs text-slate-500 mb-3">{facility.address}</p>

// //                                     <div className="space-y-2">
// //                                         <div className="flex justify-between items-center bg-slate-50 p-2 rounded border border-slate-100">
// //                                             <span className="text-xs font-medium text-slate-500">Wait Time</span>
// //                                             <span className={`text-sm font-bold ${(facility.waitTimeMins || 0) > 180 ? 'text-red-600' :
// //                                                 (facility.waitTimeMins || 0) > 60 ? 'text-yellow-600' : 'text-green-600'
// //                                                 }`}>
// //                                                 <i className="far fa-clock mr-1"></i>
// //                                                 {formatWaitTime(facility.waitTimeMins)}
// //                                             </span>
// //                                         </div>

// //                                         {facility.occupancyPercent && (
// //                                             <div className="space-y-1">
// //                                                 <div className="flex justify-between text-xs">
// //                                                     <span className="font-medium text-slate-500">Predicted Load</span>
// //                                                     <span className="font-bold text-slate-700">{facility.occupancyPercent}%</span>
// //                                                 </div>
// //                                                 <div className="w-full bg-slate-200 rounded-full h-2">
// //                                                     <div
// //                                                         className={`h-2 rounded-full ${facility.occupancyPercent > 100 ? 'bg-red-500' : facility.occupancyPercent > 80 ? 'bg-yellow-500' : 'bg-green-500'}`}
// //                                                         style={{ width: `${Math.min(facility.occupancyPercent, 100)}%` }}
// //                                                     ></div>
// //                                                 </div>
// //                                             </div>
// //                                         )}
// //                                     </div>
// //                                 </div>
// //                             </Popup>
// //                         </Marker>
// //                     );
// //                 })}
// //             </MapContainer>

// //             {/* Floating Legend / Stats Box */}
// //             <div className="absolute top-4 right-4 z-[400] bg-white/90 backdrop-blur-md p-4 rounded-xl shadow-lg border border-slate-200 pointer-events-none hidden md:block">
// //                 <h4 className="font-bold text-sm text-slate-800 mb-3 uppercase tracking-wider">Facility Status</h4>
// //                 <div className="space-y-3">
// //                     <div className="flex items-center">
// //                         <div className="w-3 h-3 rounded-full bg-red-600 border-2 border-white shadow-sm mr-2"></div>
// //                         <span className="text-xs font-medium text-slate-600 w-16">Hospital</span>
// //                         <span className="text-[10px] font-bold text-red-500 bg-red-50 px-2 py-0.5 rounded ml-2">OVER CAPACITY</span>
// //                     </div>
// //                     <div className="flex items-center">
// //                         <div className="w-3 h-3 rounded-full bg-yellow-500 border-2 border-white shadow-sm mr-2"></div>
// //                         <span className="text-xs font-medium text-slate-600 w-16">Clinic</span>
// //                         <span className="text-[10px] font-bold text-yellow-600 bg-yellow-50 px-2 py-0.5 rounded ml-2">MODERATE</span>
// //                     </div>
// //                     <div className="flex items-center">
// //                         <div className="w-3 h-3 rounded-full bg-green-600 border-2 border-white shadow-sm mr-2"></div>
// //                         <span className="text-xs font-medium text-slate-600 w-16">Pharmacy</span>
// //                         <span className="text-[10px] font-bold text-green-600 bg-green-50 px-2 py-0.5 rounded ml-2">AVAILABLE</span>
// //                     </div>
// //                 </div>
// //             </div>
// //         </div>
// //     );
// // };

// export default MapView;