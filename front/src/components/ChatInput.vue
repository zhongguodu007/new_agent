<script setup lang="ts">
import { ref } from 'vue'
import { ElInput, ElButton } from 'element-plus'

const emit = defineEmits<{
    (e: 'send-message', message: string): void
}>()

const messageInput = ref('')

const sendMessage = () => {
    if (messageInput.value.trim()) {
        emit('send-message', messageInput.value)
        messageInput.value = ''
    }
}
</script>

<template>
    <div class="chat-input-container">
        <ElInput v-model="messageInput" type="textarea" :rows="3" placeholder="输入你的消息..." @keyup.enter="sendMessage" />
        <div class="send-button-container">
            <ElButton type="primary" @click="sendMessage">
                发送
            </ElButton>
        </div>
    </div>
</template>

<style scoped>
.chat-input-container {
    padding: 15px;
    border-top: 1px solid #f0f0f0;
}

.send-button-container {
    display: flex;
    justify-content: flex-end;
    margin-top: 10px;
}
</style>