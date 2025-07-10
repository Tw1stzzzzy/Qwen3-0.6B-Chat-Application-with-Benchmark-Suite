
import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import threading
import time
import gc
import os

# 设置环境变量优化内存分配
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class QwenChatBot:
    def __init__(self):
        self.model_name = "Qwen/Qwen3-0.6B"
        self.tokenizer = None
        self.model = None
        self.conversation_count = 0
        self.max_history_length = 10  # 限制历史对话长度
        self.load_model()
    
    def clear_gpu_memory(self):
        """强制清理GPU内存"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        gc.collect()
    
    def get_gpu_memory_info(self):
        """获取GPU内存使用信息"""
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3   # GB
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
            free = total - reserved
            return {
                "allocated": allocated,
                "reserved": reserved, 
                "total": total,
                "free": free,
                "usage_percent": (reserved / total) * 100
            }
        return None
    
    def check_memory_pressure(self):
        """检查内存压力，如果内存使用过高则清理"""
        memory_info = self.get_gpu_memory_info()
        if memory_info and memory_info["usage_percent"] > 85:
            print(f"⚠️  High memory usage detected: {memory_info['usage_percent']:.1f}%")
            self.clear_gpu_memory()
            return True
        return False
    
    def load_model(self):
        """Load model and tokenizer with optimized settings"""
        print("Loading Qwen3-0.6B model with memory optimization...")
        
        # 清理现有内存
        self.clear_gpu_memory()
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,  # 使用半精度以节省内存
            device_map="auto",
            low_cpu_mem_usage=True,     # 减少CPU内存使用
            use_cache=True,             # 启用KV缓存但我们会主动管理
        )
        
        print("Model loaded successfully!")
        
        # 显示初始内存状态
        memory_info = self.get_gpu_memory_info()
        if memory_info:
            print(f"📊 Initial GPU memory: {memory_info['allocated']:.2f}GB allocated, {memory_info['free']:.2f}GB free")
    
    def trim_history(self, history):
        """修剪历史对话以节省内存"""
        if len(history) > self.max_history_length:
            # 保留最近的对话
            trimmed = history[-self.max_history_length:]
            print(f"🔄 Trimmed conversation history from {len(history)} to {len(trimmed)} entries")
            return trimmed
        return history
    
    def generate_response(self, message, history, max_tokens=256, temperature=0.7, top_p=0.8):
        """Generate response with aggressive memory management"""
        try:
            # 增加对话计数器
            self.conversation_count += 1
            
            # 每5次对话强制清理一次内存
            if self.conversation_count % 5 == 0:
                print("🧹 Performing periodic memory cleanup...")
                self.clear_gpu_memory()
            
            # 检查内存压力
            self.check_memory_pressure()
            
            # 修剪历史对话
            history = self.trim_history(history)
            
            # Build message history
            messages = []
            
            # Add chat history (限制数量)
            for user_msg, assistant_msg in history[-5:]:  # 只保留最近5轮对话
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": assistant_msg})
            
            # Add current user message
            messages.append({"role": "user", "content": message})
            
            # Use official chat template, disable thinking mode for direct response
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False  # Disable thinking mode for direct response
            )
            
            print(f"🔍 Input text length: {len(text)} chars")
            
            # 强制清理GPU缓存
            self.clear_gpu_memory()
            
            # Encode input with memory optimization
            model_inputs = self.tokenizer(
                [text], 
                return_tensors="pt", 
                truncation=True,      # 截断过长输入
                max_length=1024,      # 限制输入长度
            ).to(self.model.device)
            
            # 降低生成参数以节省内存
            generation_kwargs = {
                "max_new_tokens": min(max_tokens, 128),  # 进一步降低最大token数
                "temperature": temperature,
                "top_p": top_p,
                "do_sample": True,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
                "use_cache": False,   # 禁用缓存以节省内存
                "attention_mask": model_inputs.attention_mask,
            }
            
            print(f"🔧 Generation parameters: max_tokens={generation_kwargs['max_new_tokens']}")
            
            # 显示生成前内存状态
            memory_info = self.get_gpu_memory_info()
            if memory_info:
                print(f"📊 Pre-generation GPU memory: {memory_info['allocated']:.2f}GB used, {memory_info['free']:.2f}GB free")
            
            # Generate response with proper memory management
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=model_inputs.input_ids,
                    **generation_kwargs
                )
            
            # Extract generated tokens
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            
            # Decode response
            response = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
            
            print(f"✅ Generated response length: {len(response)} chars")
            
            # 立即清理所有临时变量
            del model_inputs, generated_ids, output_ids
            self.clear_gpu_memory()
            
            # Simple cleanup
            response = response.strip()
            
            # Provide default response if empty
            if not response:
                if "hello" in message.lower() or "hi" in message.lower():
                    response = "Hello! I'm an AI assistant, nice to meet you. How can I help you today?"
                elif any(x in message.lower() for x in ["calculate", "math", "=", "+"]):
                    response = "I can help you with calculations. Please tell me the specific equation."
                else:
                    response = "I understand your question. Is there anything else I can help you with?"
                
            return response
            
        except torch.cuda.OutOfMemoryError as e:
            # 处理OOM错误
            print(f"💥 GPU OOM Error: {str(e)}")
            
            # 强制清理所有GPU内存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                torch.cuda.synchronize()
            gc.collect()
            
            # 显示内存状态
            memory_info = self.get_gpu_memory_info()
            if memory_info:
                print(f"📊 Post-cleanup GPU memory: {memory_info['allocated']:.2f}GB used, {memory_info['free']:.2f}GB free")
            
            error_msg = f"❌ GPU内存不足。请尝试：\n1. 降低最大令牌数到64-128\n2. 点击清空对话\n3. 重启应用\n\n当前GPU使用率：{memory_info['usage_percent']:.1f}%" if memory_info else "❌ GPU内存不足，请重启应用。"
            return error_msg
        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}"
            print(error_msg)
            return error_msg
    
    def reset_conversation(self):
        """重置对话状态并清理内存"""
        self.conversation_count = 0
        self.clear_gpu_memory()
        print("🔄 Conversation reset and memory cleared")

# Create chatbot instance
chatbot = QwenChatBot()

def chat_fn(message, history, max_tokens, temperature, top_p):
    """Chat function with memory management"""
    if not message.strip():
        return history, ""
    
    # 检查历史长度，如果太长则建议清空
    if len(history) > 15:
        warning_msg = "⚠️ 对话历史较长，建议点击'清空'按钮释放内存。"
        history.append(("系统提示", warning_msg))
        return history, ""
    
    # Generate response
    response = chatbot.generate_response(message, history, max_tokens, temperature, top_p)
    
    # Update history
    history.append((message, response))
    
    return history, ""

def clear_chat():
    """Clear chat history and reset conversation"""
    chatbot.reset_conversation()
    return [], ""

def show_memory_status():
    """显示当前内存状态"""
    memory_info = chatbot.get_gpu_memory_info()
    if memory_info:
        status = f"""
        📊 **GPU内存状态**
        - 已使用: {memory_info['allocated']:.2f}GB
        - 已保留: {memory_info['reserved']:.2f}GB  
        - 总计: {memory_info['total']:.2f}GB
        - 空闲: {memory_info['free']:.2f}GB
        - 使用率: {memory_info['usage_percent']:.1f}%
        """
        return status
    return "GPU内存信息不可用"

# Create Gradio Interface
with gr.Blocks(
    title="Qwen3-0.6B Chatbot (Memory Optimized)",
    theme=gr.themes.Soft(),
    css="""
    .gradio-container {
        max-width: 900px !important;
        margin: auto !important;
    }
    """
) as demo:
    
    gr.Markdown(
        """
        # Qwen3-0.6B Chatbot (内存优化版)
        
        Welcome to the intelligent chat assistant powered by Qwen3-0.6B! 已优化GPU内存管理。
        
        > **内存优化特性**: 
        > - 自动内存清理和监控
        > - 智能历史对话修剪
        > - 降低默认参数减少内存使用
        > - OOM错误恢复机制
        """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            # 聊天界面
            chatbot_ui = gr.Chatbot(
                height=500,
                show_label=False,
                container=False,
                show_copy_button=True
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Type your message here...",
                    show_label=False,
                    scale=4,
                    container=False
                )
                send_btn = gr.Button("Send 📤", scale=1, variant="primary")
                clear_btn = gr.Button("Clear 🗑️", scale=1, variant="secondary")
        
        with gr.Column(scale=1):
            gr.Markdown("### ⚙️ 生成参数 (内存优化)")
            
            max_tokens = gr.Slider(
                minimum=32,
                maximum=256,      # 降低最大值
                value=128,        # 降低默认值
                step=16,
                label="Max Tokens",
                info="响应最大长度 (已优化)"
            )
            
            temperature = gr.Slider(
                minimum=0.1,
                maximum=1.5,      # 降低最大值
                value=0.7,
                step=0.1,
                label="Temperature",
                info="创造性控制"
            )
            
            top_p = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.8,
                step=0.05,
                label="Top-p",
                info="词汇多样性控制"
            )
            
            # 内存状态显示
            memory_status = gr.Markdown(show_memory_status())
            refresh_memory_btn = gr.Button("刷新内存状态 🔄", size="sm")
            
            gr.Markdown(
                """
                ### 💡 内存优化提示
                - **Max Tokens**: 建议使用64-128以节省内存
                - **长对话**: 超过15轮后建议清空历史
                - **OOM错误**: 降低参数或重启应用
                
                ### 🎯 推荐设置
                - 日常对话: Tokens=64, Temp=0.7
                - 创意写作: Tokens=128, Temp=1.0  
                - 问答对话: Tokens=96, Temp=0.5
                """
            )
    
    # 事件绑定
    send_btn.click(
        chat_fn,
        inputs=[msg_input, chatbot_ui, max_tokens, temperature, top_p],
        outputs=[chatbot_ui, msg_input]
    )
    
    msg_input.submit(
        chat_fn,
        inputs=[msg_input, chatbot_ui, max_tokens, temperature, top_p],
        outputs=[chatbot_ui, msg_input]
    )
    
    clear_btn.click(clear_chat, outputs=[chatbot_ui, msg_input])
    
    refresh_memory_btn.click(
        lambda: show_memory_status(),
        outputs=[memory_status]
    )

if __name__ == "__main__":
    print("🌟 Starting Qwen3-0.6B Chat Application (Memory Optimized)...")
    print("🔧 Memory optimization features enabled:")
    print("   - Expandable segments for reduced fragmentation")
    print("   - Aggressive memory cleanup")
    print("   - Lower default parameters")
    print("   - History length limiting")
    
    demo.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,
        share=False,  # Set to True to get public link
        show_error=True
    ) 