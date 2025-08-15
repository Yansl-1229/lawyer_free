import gradio as gr

# 定义获取请求信息的函数
def get_request_info(input_text, request: gr.Request):
    """获取并显示用户的IP地址和浏览器信息"""
    if request:
        print(request)
        user_agent = request.headers.get('user-agent', '未知')
        client_ip = request.client.host if request.client else '未知IP'
        session_hash = request.session_hash
        
        return f"""
        **客户端信息：**
        - IP地址：{client_ip}
        - 用户代理：{user_agent}
        - 会话ID：{session_hash}
        - 输入内容：{input_text}
        """
    return "无法获取请求信息"

# 使用Blocks创建更灵活的界面
with gr.Blocks(title="Gradio Request测试", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🌐 Gradio Request信息获取测试")
    gr.Markdown("输入任意文本，查看服务器获取到的客户端信息")
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="输入文本",
                placeholder="在这里输入一些文字...",
                lines=3
            )
            submit_btn = gr.Button("获取信息", variant="primary")
        
        with gr.Column():
            text_output = gr.Markdown(label="客户端信息")
    
    # 绑定事件
    submit_btn.click(
        fn=get_request_info,
        inputs=[text_input],
        outputs=[text_output]
    )
    
    # 也支持回车提交
    text_input.submit(
        fn=get_request_info,
        inputs=[text_input],
        outputs=[text_output]
    )

# 启动应用
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # 设置为True可以生成公网链接
        show_error=True
    )