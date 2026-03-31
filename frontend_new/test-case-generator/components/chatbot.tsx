"use client"

import { useState, useRef, useEffect } from "react"
import { MessageCircle, X, Brain, Sparkles } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

interface ChatMessage {
  role: "user" | "assistant"
  content: string
}

interface ChatbotProps {
  onSendMessage: (message: string) => Promise<void>
  chatMessages: ChatMessage[]
  isGenerating: boolean
  selectedModel?: 'gemini' | 'rag'
  onModelChange?: (model: 'gemini' | 'rag') => void
}

export function Chatbot({ onSendMessage, chatMessages, isGenerating, selectedModel = 'rag', onModelChange }: ChatbotProps) {
  const [showChatbot, setShowChatbot] = useState(false)
  const [chatMessage, setChatMessage] = useState("")
  const [chatbotWidth, setChatbotWidth] = useState(400)
  const [chatbotHeight, setChatbotHeight] = useState(500)
  const [resizeDir, setResizeDir] = useState<string | null>(null)
  const isResizing = useRef(false)
  const lastMousePosition = useRef({ x: 0, y: 0 })
  
  const minWidth = 320
  const minHeight = 350
  const maxWidth = 700
  const maxHeight = 800

  function handleResizeMouseDown(dir: string) {
    isResizing.current = true
    setResizeDir(dir)
    document.body.style.userSelect = 'none'
  }

  const handleSendMessage = async () => {
    if (!chatMessage.trim() || isGenerating) return
    const message = chatMessage.trim()
    setChatMessage("")
    await onSendMessage(message)
  }

  const handleModelChange = (model: 'gemini' | 'rag') => {
    if (onModelChange) {
      onModelChange(model)
    }
  }

  useEffect(() => {
    function handleMouseMove(e: MouseEvent) {
      if (!isResizing.current || !resizeDir) return;
      if (resizeDir.includes('e')) {
        setChatbotWidth((w) => Math.max(minWidth, Math.min(maxWidth, w + e.movementX)));
      }
      if (resizeDir.includes('s')) {
        setChatbotHeight((h) => Math.max(minHeight, Math.min(maxHeight, h + e.movementY)));
      }
      if (resizeDir.includes('w')) {
        setChatbotWidth((w) => Math.max(minWidth, Math.min(maxWidth, w - e.movementX)));
      }
      if (resizeDir.includes('n')) {
        setChatbotHeight((h) => Math.max(minHeight, Math.min(maxHeight, h - e.movementY)));
      }
    }
    function handleMouseUp() {
      isResizing.current = false;
      setResizeDir(null);
      document.body.style.userSelect = '';
    }
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);
    return () => {
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [resizeDir]);

  return (
    <div
      className={!showChatbot ? "fixed bottom-6 right-6 z-50" : undefined}
      style={!showChatbot ? { width: 0, height: 0 } : undefined}
    >
      {!showChatbot ? (
        <Button
          onClick={() => setShowChatbot(true)}
          size="lg"
          className="rounded-full h-14 w-14 shadow-2xl hover:shadow-3xl transition-all duration-300 bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 fixed bottom-6 right-6 z-50 transform hover:scale-110"
        >
          <MessageCircle className="h-6 w-6" />
        </Button>
      ) : (
        <div
          className="fixed bottom-6 right-6 z-50 flex flex-col h-full bg-white border-0 rounded-2xl shadow-2xl overflow-hidden"
          style={{ width: chatbotWidth, height: chatbotHeight, minWidth, minHeight, maxWidth, maxHeight, boxShadow: '0 20px 60px rgba(0,0,0,0.25)' }}
        >
          {/* Enhanced Header */}
          <div className="flex flex-col px-6 py-4 bg-gradient-to-r from-blue-500 to-indigo-600 border-b border-blue-400">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-white/20 rounded-full flex items-center justify-center">
                  <MessageCircle className="h-4 w-4 text-white" />
                </div>
                <span className="font-bold text-white text-base">AI Test Case Assistant</span>
              </div>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowChatbot(false)}
                className="text-white hover:bg-white/20 rounded-full p-2"
              >
                <X className="h-5 w-5" />
              </Button>
            </div>
            {/* Model Selection */}
            <div className="flex gap-2 mt-3">
              <Button
                size="sm"
                variant={selectedModel === 'rag' ? 'secondary' : 'ghost'}
                onClick={() => handleModelChange('rag')}
                className={`flex-1 ${selectedModel === 'rag' ? 'bg-white text-blue-600' : 'text-white hover:bg-white/20'}`}
              >
                <Sparkles className="h-4 w-4 mr-2" />
                RAG Mode
              </Button>
              <Button
                size="sm"
                variant={selectedModel === 'gemini' ? 'secondary' : 'ghost'}
                onClick={() => handleModelChange('gemini')}
                className={`flex-1 ${selectedModel === 'gemini' ? 'bg-white text-blue-600' : 'text-white hover:bg-white/20'}`}
              >
                <Brain className="h-4 w-4 mr-2" />
                Gemini Mode
              </Button>
            </div>
          </div>
          
          {/* Enhanced Chat area */}
          <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4 bg-gradient-to-br from-gray-50 to-blue-50/30" style={{scrollbarWidth:'thin'}}>
            {chatMessages.map((message, index) => (
              <div key={index} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={
                    message.role === "user"
                      ? "bg-gradient-to-r from-blue-500 to-indigo-600 text-white rounded-2xl rounded-br-sm px-4 py-3 max-w-[80%] shadow-lg"
                      : "bg-white text-gray-800 rounded-2xl rounded-bl-sm px-4 py-3 max-w-[80%] shadow-lg border border-gray-200"
                  }
                  style={{ wordBreak: 'break-word', whiteSpace: 'pre-wrap' }}
                >
                  {message.role === "assistant" ? (
                    <pre className="text-sm whitespace-pre-wrap" style={{background:'none',margin:0,padding:0,border:'none'}}>
                      {message.content && message.content.toString().trim() ? message.content : '[No output from LLM]'}
                    </pre>
                  ) : (
                    <span className="text-sm whitespace-pre-wrap">{message.content}</span>
                  )}
                </div>
              </div>
            ))}
            {isGenerating && (
              <div className="flex justify-start">
                <div className="inline-flex items-center gap-2 p-3 rounded-xl bg-white border border-gray-200 text-slate-600 text-sm shadow-lg">
                  <div className="flex gap-1">
                    <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.3s]"></div>
                    <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce [animation-delay:-0.15s]"></div>
                    <div className="w-1.5 h-1.5 bg-blue-500 rounded-full animate-bounce"></div>
                  </div>
                  <span>Generating response...</span>
                </div>
              </div>
            )}
          </div>

          {/* Enhanced Input area */}
          <div className="p-4 bg-white border-t border-gray-100">
            <div className="flex gap-2">
              <Input
                placeholder={selectedModel === 'rag' ? "Describe your user story..." : "Ask me anything about testing..."}
                value={chatMessage}
                onChange={(e) => setChatMessage(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && !isGenerating && handleSendMessage()}
                disabled={isGenerating}
                className="border-2 border-gray-200 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 rounded-xl flex-1 text-sm shadow-sm"
              />
              <Button
                size="sm"
                onClick={handleSendMessage}
                disabled={isGenerating || !chatMessage.trim()}
                className="bg-gradient-to-r from-blue-500 to-indigo-600 hover:from-blue-600 hover:to-indigo-700 text-white rounded-xl px-6 py-2 text-sm font-semibold shadow-lg transition-all duration-200 transform hover:scale-105 disabled:transform-none"
              >
                {selectedModel === 'rag' ? 'Generate' : 'Ask'}
              </Button>
            </div>
          </div>
          
          {/* Resize handles: corners and sides */}
          {/* Corners */}
          <div onMouseDown={() => handleResizeMouseDown('nw')} className="absolute top-0 left-0 w-3 h-3 cursor-nwse-resize z-10" />
          <div onMouseDown={() => handleResizeMouseDown('ne')} className="absolute top-0 right-0 w-3 h-3 cursor-nesw-resize z-10" />
          <div onMouseDown={() => handleResizeMouseDown('sw')} className="absolute bottom-0 left-0 w-3 h-3 cursor-nesw-resize z-10" />
          <div onMouseDown={() => handleResizeMouseDown('se')} className="absolute bottom-0 right-0 w-3 h-3 cursor-nwse-resize z-10" />
          {/* Sides */}
          <div onMouseDown={() => handleResizeMouseDown('n')} className="absolute top-0 left-3 right-3 h-2 cursor-ns-resize z-10" />
          <div onMouseDown={() => handleResizeMouseDown('s')} className="absolute bottom-0 left-3 right-3 h-2 cursor-ns-resize z-10" />
          <div onMouseDown={() => handleResizeMouseDown('e')} className="absolute top-3 bottom-3 right-0 w-2 cursor-ew-resize z-10" />
          <div onMouseDown={() => handleResizeMouseDown('w')} className="absolute top-3 bottom-3 left-0 w-2 cursor-ew-resize z-10" />
        </div>
      )}
    </div>
  )
} 