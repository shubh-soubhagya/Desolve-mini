// import { useContext } from "react";
// import { AppContext } from "../context/AppContext";
// import { marked } from "marked";

// export default function SummaryPage({ onStartChat }) {
//   const { summary } = useContext(AppContext);

//   return (
//     <div className="max-w-4xl mx-auto p-6 mt-6 bg-white rounded shadow">
//       <h1 className="text-3xl font-semibold mb-4">Repository Summary</h1>

//       <div
//         className="prose max-w-none bg-gray-50 p-4 rounded"
//         dangerouslySetInnerHTML={{ __html: marked(summary || "") }}
//       />

//       <button
//         onClick={onStartChat}
//         className="mt-6 px-6 py-3 bg-blue-600 text-white rounded"
//       >
//         Start Chat
//       </button>
//     </div>
//   );
// }


import { useContext } from "react";
import { AppContext } from "../context/AppContext";
import { marked } from "marked";
import Navbar from "../components/Navbar"; // Assuming path is ./Navbar based on previous context
import { FiFileText, FiMessageSquare, FiArrowRight, FiZap } from "react-icons/fi";

export default function SummaryPage({ onStartChat }) {
  const { summary } = useContext(AppContext);

  return (
    <div className="min-h-screen w-full fixed inset-0 z-50 flex flex-col font-sans">
      
      {/* --- Background Video --- */}
      <video 
        autoPlay 
        loop 
        muted 
        className="absolute inset-0 w-full h-full object-cover -z-20"
      >
        <source src="/video2.mp4" type="video/mp4" />
      </video>

      {/* --- Navbar --- */}
      <div className="z-50">
        <Navbar />
      </div>

      {/* --- Main Content Area --- */}
      <div className="flex-1 overflow-y-auto p-4 md:p-6 custom-scrollbar flex justify-center">
        
        {/* MASTER GLASS CARD 
           - bg-white/80: Increased opacity slightly since we removed the inner div, 
             ensuring text is readable while keeping the glass effect.
           - backdrop-blur-2xl: Strong blur to smooth out video motion behind text.
        */}
        <div className="w-full max-w-5xl bg-white/80 backdrop-blur-2xl border border-white/60 shadow-2xl rounded-2xl overflow-hidden flex flex-col animate-in fade-in zoom-in-95 duration-500">
            
          {/* --- Narrow Header --- */}
          <div className="flex justify-between items-center py-3 px-6 border-b border-gray-200/50 bg-white/40 backdrop-blur-sm sticky top-0 z-10">
            <div className="flex items-center gap-3">
              <div className="p-1.5 bg-blue-50 rounded-lg text-blue-600 border border-blue-100">
                <FiFileText size={18} />
              </div>
              <div>
                <h1 className="text-lg font-bold text-gray-900 tracking-tight leading-none">
                  Repository Summary
                </h1>
                <p className="text-[10px] text-gray-500 font-bold uppercase tracking-widest">
                  AI Generated
                </p>
              </div>
            </div>

            {/* Header Button: Minimalist Glass Pill */}
            <button 
              onClick={onStartChat} 
              className="group flex items-center gap-2 px-4 py-1.5 bg-white/50 hover:bg-white border border-gray-200 hover:border-blue-300 text-gray-700 hover:text-blue-600 rounded-full text-sm font-semibold transition-all shadow-sm hover:shadow-md"
            >
              <span>Quick Chat</span>
              <FiMessageSquare className="w-4 h-4 group-hover:scale-110 transition-transform" />
            </button>
          </div>

          {/* --- Content Section (No Inner Div) --- */}
          <div className="flex-1 p-8 md:p-10 overflow-y-auto">
            <article 
              className="prose prose-lg prose-slate max-w-none 
              prose-headings:font-bold prose-headings:text-gray-900 prose-headings:tracking-tight
              prose-p:text-gray-800 prose-p:leading-relaxed prose-p:font-medium
              prose-li:text-gray-800
              prose-strong:text-blue-700
              prose-a:text-blue-600 prose-a:font-semibold prose-a:no-underline hover:prose-a:underline
              prose-code:bg-white/80 prose-code:text-pink-600 prose-code:px-1.5 prose-code:py-0.5 prose-code:rounded-md prose-code:border prose-code:border-gray-200/50 prose-code:font-semibold prose-code:shadow-sm
              prose-pre:bg-gray-900 prose-pre:text-gray-100 prose-pre:shadow-xl prose-pre:rounded-xl prose-pre:border prose-pre:border-gray-800"
              dangerouslySetInnerHTML={{ __html: marked(summary || "*No summary available. Please check the repository URL.*") }}
            />
          </div>

          {/* --- Footer Section --- */}
          <div className="py-4 px-6 bg-white/60 border-t border-gray-200/50 flex justify-between items-center backdrop-blur-xl">
             <div className="flex items-center gap-2 text-xs font-semibold text-gray-500 uppercase tracking-wider">
               <FiZap className="text-amber-500 text-sm" />
               <span>Analysis Ready</span>
             </div>

             {/* Footer Button: High Impact Gradient */}
             <button 
              onClick={onStartChat} 
              className="group relative flex items-center gap-3 bg-gradient-to-r from-gray-900 to-gray-800 hover:from-blue-600 hover:to-blue-700 text-white text-base font-bold py-2.5 px-8 rounded-xl shadow-lg hover:shadow-blue-500/30 transition-all duration-300 transform active:scale-95"
            >
              <span>Start Workspace</span>
              <FiArrowRight className="group-hover:translate-x-1 transition-transform" />
            </button>
          </div>

        </div>
      </div>
    </div>
  );
}