// import { useContext } from "react";
// import { AppContext } from "../context/AppContext";
// import { FiMenu } from "react-icons/fi";

// export default function Navbar() {
//   const { setSidebarOpen } = useContext(AppContext);

//   return (
//     <nav className="bg-white shadow px-4 py-3 flex items-center justify-between">
//       <h1 className="text-xl font-semibold">DissolveAI</h1>

//       <button
//         onClick={() => setSidebarOpen(true)}
//         className="text-2xl p-2"
//       >
//         <FiMenu />
//       </button>
//     </nav>
//   );
// }


import { useContext } from "react";
import { AppContext } from "../context/AppContext";
import { FiMenu, FiSettings, FiBell } from "react-icons/fi";

export default function Navbar() {
  const { setSidebarOpen } = useContext(AppContext);

  return (
    <nav className="sticky top-0 z-30 w-full bg-white/60 backdrop-blur-2xl border-b border-white/40 shadow-sm px-8 py-4 flex items-center justify-between transition-all duration-300">
      
      {/* Left: Brand Identity */}
      <div className="flex items-center gap-4">
        <div className="w-10 h-10 bg-gradient-to-br from-gray-900 to-gray-700 rounded-xl flex items-center justify-center shadow-lg text-white font-bold text-lg">
          D
        </div>
        <div className="hidden md:block">
          <h1 className="text-xl font-bold text-gray-900 tracking-tight leading-none">
            DissolveAI
          </h1>
          <p className="text-[10px] font-semibold text-gray-500 uppercase tracking-widest mt-0.5">
            Enterprise Edition
          </p>
        </div>
      </div>

      {/* Right: Actions */}
      <div className="flex items-center gap-4">
        
        {/* Optional: Notification Icon (adds to professional feel) */}
        <button className="p-2.5 text-gray-500 hover:text-gray-900 hover:bg-white/50 rounded-xl transition-colors">
          <FiBell size={20} />
        </button>

        {/* Configuration Trigger */}
        <button
          onClick={() => setSidebarOpen(true)}
          className="group flex items-center gap-3 px-5 py-2.5 bg-white/40 hover:bg-white/60 border border-white/20 backdrop-blur-md rounded-xl text-gray-800 font-medium transition-all shadow-sm hover:shadow-md active:scale-95"
        >
          <FiSettings className="w-5 h-5 text-gray-600 group-hover:rotate-90 transition-transform duration-500" />
          <span className="hidden sm:inline">Configuration Settings</span>
        </button>
      </div>
    </nav>
  );
}