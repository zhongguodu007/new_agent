<script setup lang="ts">
import { computed, ref } from 'vue'
import { ElAvatar } from 'element-plus'

interface Message {
    id: number
    content: string
    role: 'user' | 'ai'
    timestamp?: string
    source?: string
}

const props = defineProps<{
    message: Message
}>()

const isUserMessage = computed(() => props.message.role === 'user')
const avatarSrc = computed(() =>
    isUserMessage.value
        ? '/img/user-avatar.png'  // 可以替换为实际用户头像
        : '/img/ai-avatar.png'    // 可以替换为AI头像
)

// 流式输出相关
const isCompleted = ref(props.message.role !== 'ai') // 用户消息立即完成，AI消息需要模拟流式输出
const displayedContent = ref(props.message.role === 'user' ? props.message.content : '')
const typingInterval = ref(null as any)

// 模拟打字机效果
if (props.message.role === 'ai') {
    let index = 0
    const content = props.message.content
    
    typingInterval.value = setInterval(() => {
        if (index < content.length) {
            displayedContent.value += content[index]
            index++
        } else {
            clearInterval(typingInterval.value)
            isCompleted.value = true
        }
    }, 30) // 调整速度
}

// 组件卸载时清除定时器
import { onBeforeUnmount } from 'vue'
onBeforeUnmount(() => {
    if (typingInterval.value) {
        clearInterval(typingInterval.value)
    }
})
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
        <div class="message-content" :class="{ 'typing': !isCompleted }">
            <div class="message-text">{{ displayedContent }}</div>
            <div v-if="!isCompleted" class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
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
    padding: 12px 16px;
    border-radius: 10px;
    box-shadow: 0 1px 4px rgba(0, 0, 0, 0.1);
    position: relative;
    word-wrap: break-word;
    white-space: pre-wrap;
    line-height: 1.5;
}

.message-text {
    font-size: 14px;
}

.user-message {
    flex-direction: row-reverse;
}

.user-message .message-content {
    background-color: #e6f3ff;
    border-top-right-radius: 0;
}

.ai-message .message-content {
    background-color: #f5f5f5;
    border-top-left-radius: 0;
}

.user-avatar, .ai-avatar {
    flex-shrink: 0;
}

/* 打字机效果相关样式 */
.typing-indicator {
    display: inline-flex;
    align-items: center;
    margin-top: 5px;
}

.typing-indicator span {
    height: 8px;
    width: 8px;
    background-color: #409eff;
    border-radius: 50%;
    display: inline-block;
    margin: 0 2px;
    opacity: 0.6;
    animation: typing 1.2s infinite;
}

.typing-indicator span:nth-child(2) {
    animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
    animation-delay: 0.4s;
}

@keyframes typing {
    0%, 60%, 100% {
        transform: translateY(0);
    }
    30% {
        transform: translateY(-5px);
    }
}
</style>