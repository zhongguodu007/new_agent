<script setup lang="ts">
import { computed } from 'vue'
import { ElAvatar } from 'element-plus'

interface Message {
    id: number
    content: string
    role: 'user' | 'ai'
    timestamp?: string
}

const props = defineProps<{
    message: Message
}>()

const isUserMessage = computed(() => props.message.role === 'user')
const avatarSrc = computed(() =>
    isUserMessage.value
        ? 'https://placekitten.com/40/40'
        : 'https://placekitten.com/41/41'
)
</script>

<template>
    <div class="chat-message" :class="{
        'user-message': isUserMessage,
        'ai-message': !isUserMessage
    }">
        <ElAvatar :src="avatarSrc" :class="{
            'user-avatar': isUserMessage,
            'ai-avatar': !isUserMessage
        }" />
        <div class="message-content">
            {{ message.content }}
        </div>
    </div>
</template>

<style scoped>
.chat-message {
    display: flex;
    align-items: flex-start;
    margin-bottom: 15px;
    gap: 10px;
}

.message-content {
    max-width: 70%;
    padding: 10px;
    border-radius: 10px;
}

.user-message {
    flex-direction: row-reverse;
}

.user-message .message-content {
    background-color: #e6f3ff;
}

.ai-message .message-content {
    background-color: #f0f0f0;
}
</style>