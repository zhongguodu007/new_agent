<script setup lang="ts">
import { ref } from 'vue'
import { ElInput, ElButton } from 'element-plus'
import { Position } from '@element-plus/icons-vue'

const emit = defineEmits<{
    (e: 'send-message', message: string): void
}>()

const messageInput = ref('')
const isTyping = ref(false)

// 处理键盘事件
const handleKeydown = (e: KeyboardEvent) => {
    // 如果按下Enter键且没有按下Shift键，发送消息
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault() // 阻止默认的换行行为
        sendMessage()
    }
    // 如果按下Shift+Enter，允许换行
    // 不需要额外处理，因为这是textarea的默认行为
}

const sendMessage = () => {
    if (messageInput.value.trim()) {
        emit('send-message', messageInput.value)
        messageInput.value = ''
    }
}
</script>

<template>
    <div class="chat-input-container">
        <ElInput 
            v-model="messageInput" 
            type="textarea" 
            :rows="3" 
            placeholder="输入你的消息... (按Enter发送，Shift+Enter换行)" 
            @keydown="handleKeydown"
            resize="none"
        />
        <div class="send-button-container">
            <ElButton type="primary" @click="sendMessage">
                <el-icon><Position /></el-icon>
                发送
            </ElButton>
        </div>
    </div>
</template>

<style scoped>
.chat-input-container {
    padding: 15px;
    border-top: 1px solid #f0f0f0;
    background-color: #fff;
}

.send-button-container {
    display: flex;
    justify-content: flex-end;
    margin-top: 10px;
}

:deep(.el-textarea__inner) {
    resize: none;
    border: 1px solid #dcdfe6;
    border-radius: 8px;
    padding: 12px;
    transition: all 0.3s;
    font-size: 14px;
    line-height: 1.5;
}

:deep(.el-textarea__inner:focus) {
    border-color: #409eff;
    box-shadow: 0 0 0 2px rgba(64, 158, 255, 0.2);
}
</style>