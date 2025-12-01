// import { useContext, useState } from "react";
// import { AppContext } from "../context/AppContext";
// import backend from "../api/api";
// import MarkdownBubble from "./MarkdownBubble";

// export default function ChatUI() {
//   const {
//     chatMessages, setChatMessages,
//     selectedIssue, selectedModel
//   } = useContext(AppContext);

//   const [input, setInput] = useState("");
//   const [loading, setLoading] = useState(false);

//   async function sendMessage() {
//     if (!input.trim()) return;

//     setChatMessages([...chatMessages,
//       { role: "user", text: input }
//     ]);

//     const userText = input;
//     setInput("");
//     setLoading(true);

//     try {
//       const res = await backend.askAI({
//         issue_id: selectedIssue,
//         prompt: userText,
//         model: selectedModel
//       });

//       setChatMessages(m => [
//         ...m,
//         { role: "assistant", text: res.data.response }
//       ]);

//     } catch {
//       setChatMessages(m => [...m, {
//         role: "assistant",
//         text: "Error occurred"
//       }]);
//     }

//     setLoading(false);
//   }

//   return (
//     <div className="relative h-full">
//       {/* Video Background */}
//       <div className="absolute inset-0 z-0">
//         <video
//           autoPlay
//           muted
//           loop
//           className="w-full h-full object-cover"
//         >
//           <source src="/video2.mp4" type="video/mp4" />
//           {/* Fallback image if video doesn't load */}
//           <div className="w-full h-full bg-gradient-to-br from-blue-50 to-purple-50"></div>
//         </video>
//         {/* Overlay to improve text readability */}
//         <div className="absolute inset-0 bg-black/10 backdrop-blur-[1px]"></div>
//       </div>

//       {/* Chat Interface Container */}
//       <div className="relative z-10 h-full flex justify-center">
//         <div className="w-full max-w-6xl flex flex-col h-full px-4">
          
//           {/* Chat log */}
//           <div className="flex-1 overflow-y-auto py-6 space-y-4">
//             {chatMessages.length === 0 && (
//               <div className="text-center mt-20">
//                 <div className="bg-white/70 backdrop-blur-md border border-white/40 rounded-2xl p-8 max-w-md mx-auto shadow-xl">
//                   <h3 className="text-xl font-semibold text-gray-700 mb-2">Welcome to DissolveAI</h3>
//                   <p className="text-gray-600">Start a conversation by typing a message below</p>
//                 </div>
//               </div>
//             )}
            
//             {chatMessages.map((m, i) => (
//               <div 
//                 key={i} 
//                 className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
//               >
//                 <div className={`max-w-[85%] ${
//                   m.role === 'user' 
//                     ? 'bg-blue-600/90 backdrop-blur-md text-white shadow-lg' 
//                     : 'bg-white/90 backdrop-blur-md border border-white/40 shadow-lg'
//                 } rounded-2xl p-1`}>
//                   <MarkdownBubble text={m.text} mine={m.role === "user"} />
//                 </div>
//               </div>
//             ))}

//             {loading && (
//               <div className="flex justify-start">
//                 <div className="bg-white/90 backdrop-blur-md border border-white/40 rounded-2xl p-5 shadow-lg max-w-[85%]">
//                   <div className="flex items-center space-x-2">
//                     <div className="flex space-x-1">
//                       <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
//                       <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
//                       <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
//                     </div>
//                     <span className="text-gray-600 text-sm font-medium">Thinking...</span>
//                   </div>
//                 </div>
//               </div>
//             )}
//           </div>

//           {/* Input Area */}
//           <div className="pb-6 pt-4">
//             <div className="flex gap-3 bg-white/90 backdrop-blur-xl border border-white/50 rounded-2xl p-1 shadow-2xl">
//               <input
//                 value={input}
//                 onChange={(e) => setInput(e.target.value)}
//                 onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
//                 className="flex-1 bg-transparent border-none outline-none px-4 py-3 text-gray-800 placeholder-gray-500 text-lg resize-none"
//                 placeholder="Message DissolveAI..."
//               />

//               <button
//                 onClick={sendMessage}
//                 disabled={loading || !input.trim()}
//                 className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 backdrop-blur-sm text-white px-8 py-3 rounded-xl font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl transform hover:scale-105 disabled:hover:scale-100"
//               >
//                 Send
//               </button>
//             </div>
            
//             {/* Subtle hint text */}
//             <div className="text-center mt-3">
//               <p className="text-white/80 text-sm backdrop-blur-sm bg-black/20 rounded-full px-4 py-1 inline-block">
//                 Press Enter to send • Shift + Enter for new line
//               </p>
//             </div>
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// }


import { useContext, useState } from "react";
import { AppContext } from "../context/AppContext";
import backend from "../api/api";
import MarkdownBubble from "./MarkdownBubble";

export default function ChatUI() {
  const {
    chatMessages, setChatMessages,
    selectedIssue, selectedModel
  } = useContext(AppContext);

  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function sendMessage() {
    if (!input.trim()) return;

    setChatMessages([...chatMessages,
      { role: "user", text: input }
    ]);

    const userText = input;
    setInput("");
    setLoading(true);

    try {
      const res = await backend.askAI({
        issue_id: selectedIssue,
        prompt: userText,
        model: selectedModel
      });

      setChatMessages(m => [
        ...m,
        { role: "assistant", text: res.data.response }
      ]);

    } catch {
      setChatMessages(m => [...m, {
        role: "assistant",
        text: "Error occurred"
      }]);
    }

    setLoading(false);
  }

  return (
    <div className="relative h-full">
      {/* Video Background */}
      <div className="absolute inset-0 z-0">
        <video
          autoPlay
          muted
          loop
          className="w-full h-full object-cover"
        >
          <source src="/video2.mp4" type="video/mp4" />
          <div className="w-full h-full bg-gradient-to-br from-blue-50 to-purple-50"></div>
        </video>
        <div className="absolute inset-0 bg-black/10 backdrop-blur-[1px]"></div>
      </div>

      {/* Chat Interface Container */}
      <div className="relative z-10 h-full flex justify-center">
        <div className="w-full max-w-6xl flex flex-col h-full px-4">
          
          {/* Chat log */}
          <div className="flex-1 overflow-y-auto py-6 space-y-4 transparent-scrollbar">
            {chatMessages.length === 0 && (
              <div className="text-center mt-20">
                <div className="bg-white/70 backdrop-blur-md border border-white/40 rounded-2xl p-8 max-w-md mx-auto shadow-xl">
                  <h3 className="text-xl font-semibold text-gray-700 mb-2">Welcome to DissolveAI</h3>
                  <p className="text-gray-600">Start a conversation by typing a message below</p>
                </div>
              </div>
            )}
            
            {chatMessages.map((m, i) => (
              <div 
                key={i} 
                className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div className={`max-w-[85%] ${
                  m.role === 'user' 
                    ? 'bg-blue-600/90 backdrop-blur-md text-white shadow-lg' 
                    : 'bg-white/90 backdrop-blur-md border border-white/40 shadow-lg'
                } rounded-2xl px-1 py-1`}>
                  <MarkdownBubble text={m.text} mine={m.role === "user"} />
                </div>
              </div>
            ))}

            {loading && (
              <div className="flex justify-start">
                <div className="bg-white/90 backdrop-blur-md border border-white/40 rounded-2xl p-5 shadow-lg max-w-[85%]">
                  <div className="flex items-center space-x-2">
                    <div className="flex space-x-1">
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                      <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    </div>
                    <span className="text-gray-600 text-sm font-medium">Thinking...</span>
                  </div>
                </div>
              </div>
            )}
          </div>

          {/* Input Area */}
          <div className="pb-6 pt-4">
            <div className="flex gap-3 bg-white/90 backdrop-blur-xl border border-white/50 rounded-2xl p-1 shadow-2xl">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={(e) => e.key === 'Enter' && !e.shiftKey && sendMessage()}
                className="flex-1 bg-transparent border-none outline-none px-4 py-3 text-gray-800 placeholder-gray-500 text-lg resize-none"
                placeholder="Message DissolveAI..."
              />

              <button
                onClick={sendMessage}
                disabled={loading || !input.trim()}
                className="bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 backdrop-blur-sm text-white px-8 py-3 rounded-xl font-semibold transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed shadow-lg hover:shadow-xl transform hover:scale-105 disabled:hover:scale-100"
              >
                Send
              </button>
            </div>
            
            <div className="text-center mt-3">
              <p className="text-white/80 text-sm backdrop-blur-sm bg-black/20 rounded-full px-4 py-1 inline-block">
                Press Enter to send • Shift + Enter for new line
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
