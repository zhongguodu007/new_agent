<script setup lang="ts">
import { ref, onMounted, nextTick, watch } from 'vue'
import ChatMessage from '../components/ChatMessage.vue'
import ChatInput from '../components/ChatInput.vue'
import { ElMessage } from 'element-plus'

interface Message {
    id: number
    content: string
    role: 'user' | 'ai'
    timestamp?: string
    source?: string
}

interface Conversation {
    id: string
    title: string
    messages: Message[]
    createdAt: Date
}

const conversations = ref<Conversation[]>([])
const activeConversationId = ref('')
const messages = ref<Message[]>([])
const messagesContainer = ref<HTMLElement | null>(null)

// 滚动到底部
const scrollToBottom = async () => {
    await nextTick()
    if (messagesContainer.value) {
        messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
    }
}

// 监听消息变化，自动滚动到底部
watch(() => messages.value.length, scrollToBottom)

// 创建新的对话
const createNewConversation = () => {
    const id = Date.now().toString()
    const newConversation = {
        id,
        title: `新对话 ${conversations.value.length + 1}`,
        messages: [
            {
                id: 1,
                content: '你好！有什么可以帮你的吗？',
                role: 'ai' as const,
                timestamp: new Date().toLocaleString()
            }
        ],
        createdAt: new Date()
    }
    
    conversations.value.push(newConversation)
    setActiveConversation(id)
}

// 设置当前活跃对话
const setActiveConversation = (id: string) => {
    activeConversationId.value = id
    const conversation = conversations.value.find(c => c.id === id)
    if (conversation) {
        messages.value = conversation.messages
        scrollToBottom()
    }
}

// 删除对话
const deleteConversation = (id: string) => {
    const index = conversations.value.findIndex(c => c.id === id)
    if (index !== -1) {
        conversations.value.splice(index, 1)
        if (conversations.value.length > 0) {
            setActiveConversation(conversations.value[0].id)
        } else {
            createNewConversation()
        }
    }
}

const selectedFunction = ref('')
const functionOptions = [
    { value: 'web', label: '全网搜索' },
    { value: 'rag', label: '本地知识库搜索' },
    { value: 'cs', label: '计算机专业知识检索' }
]

const generateMockAIResponse = (userMessage: string, functionType: string) => {
    let responsePrefix = ''
    
    switch(functionType) {
        case 'web':
            responsePrefix = '[通过全网搜索] '
            break
        case 'rag':
            responsePrefix = '[通过本地知识库] '
            break
        case 'cs':
            responsePrefix = '[通过计算机专业知识库] '
            break
        default:
            responsePrefix = ''
    }

    const mockResponses = [
        `${responsePrefix}我收到了你的消息：${userMessage}。听起来很有趣！`,
        `${responsePrefix}关于"${userMessage}"，我有一些想法想和你分享。`,
        `${responsePrefix}非常感谢你分享"${userMessage}"。这确实是一个有趣的话题。`,
        `${responsePrefix}我对你说的"${userMessage}"很感兴趣，能详细告诉我更多吗？`
    ]

    return mockResponses[Math.floor(Math.random() * mockResponses.length)]
}

const handleSendMessage = (message: string) => {
    if (!activeConversationId.value) {
        createNewConversation()
    }

    const userMessage: Message = {
        id: messages.value.length + 1,
        content: message,
        role: 'user',
        timestamp: new Date().toLocaleString()
    }

    messages.value.push(userMessage)
    scrollToBottom()
    
    // 更新对话中的消息
    const conversationIndex = conversations.value.findIndex(c => c.id === activeConversationId.value)
    if (conversationIndex !== -1) {
        conversations.value[conversationIndex].messages = messages.value
        
        // 如果是第一条消息，更新对话标题
        if (conversations.value[conversationIndex].messages.length === 2) {
            conversations.value[conversationIndex].title = message.length > 15 
                ? message.substring(0, 15) + '...' 
                : message
        }
    }

    // 根据选择的功能生成回复
    const functionType = selectedFunction.value || 'rag' // 默认本地知识库
    
    // Simulate AI response
    setTimeout(() => {
        const aiResponse: Message = {
            id: messages.value.length + 1,
            content: generateMockAIResponse(message, functionType),
            role: 'ai',
            timestamp: new Date().toLocaleString(),
            source: functionType
        }
        messages.value.push(aiResponse)
        scrollToBottom()
        
        // 同步更新对话中的消息
        if (conversationIndex !== -1) {
            conversations.value[conversationIndex].messages = messages.value
        }
    }, 1000)
    
    // 使用后重置选择的功能
    selectedFunction.value = ''
}

// 初始创建一个对话
onMounted(() => {
    createNewConversation()
    scrollToBottom()
})
</script>

<template>
    <div class="chat-page">
        <aside class="chat-sidebar">
            <div class="sidebar-header">
                <h3>对话列表</h3>
                <el-button type="primary" @click="createNewConversation" size="small">
                    新建对话
                </el-button>
            </div>
            <div class="conversation-list">
                <div 
                    v-for="conv in conversations" 
                    :key="conv.id" 
                    class="conversation-item"
                    :class="{ 'active': conv.id === activeConversationId }"
                    @click="setActiveConversation(conv.id)"
                >
                    <div class="conversation-title">{{ conv.title }}</div>
                    <div class="conversation-time">{{ new Date(conv.createdAt).toLocaleDateString() }}</div>
                    <el-button 
                        class="delete-btn" 
                        type="text"
                        @click.stop="deleteConversation(conv.id)"
                    >
                        删除
                    </el-button>
                </div>
            </div>
        </aside>

        <main class="chat-main">
            <div class="chat-container">
                <div class="chat-messages" ref="messagesContainer">
                    <ChatMessage v-for="message in messages" :key="message.id" :message="message" />
                </div>
                
                <div class="chat-function-selector">
                    <el-radio-group v-model="selectedFunction" size="small">
                        <el-radio-button v-for="option in functionOptions" :key="option.value" :label="option.value">
                            {{ option.label }}
                        </el-radio-button>
                    </el-radio-group>
                </div>
                
                <ChatInput @send-message="handleSendMessage" />
            </div>
        </main>
    </div>
</template>

<style scoped>
.chat-page {
    display: flex;
    height: calc(100vh - 60px);
}

.chat-sidebar {
    width: 250px;
    border-right: 1px solid #eaeaea;
    background-color: #fff;
    display: flex;
    flex-direction: column;
}

.sidebar-header {
    padding: 20px;
    border-bottom: 1px solid #eaeaea;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.sidebar-header h3 {
    margin: 0;
    font-size: 16px;
}

.conversation-list {
    flex: 1;
    overflow-y: auto;
}

.conversation-item {
    padding: 15px 20px;
    border-bottom: 1px solid #f0f0f0;
    cursor: pointer;
    position: relative;
    transition: all 0.3s;
}

.conversation-item:hover {
    background-color: #f5f7fa;
}

.conversation-item.active {
    background-color: #ecf5ff;
}

.conversation-title {
    font-weight: 500;
    margin-bottom: 5px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

.conversation-time {
    font-size: 12px;
    color: #909399;
}

.delete-btn {
    position: absolute;
    right: 10px;
    top: 50%;
    transform: translateY(-50%);
    opacity: 0;
    transition: opacity 0.3s;
    color: #f56c6c;
}

.conversation-item:hover .delete-btn {
    opacity: 1;
}

.chat-main {
    flex: 1;
    background-color: #f5f7fa;
    display: flex;
    flex-direction: column;
}

.chat-container {
    max-width: 800px;
    margin: 0 auto;
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
}

.chat-messages {
    flex-grow: 1;
    overflow-y: auto;
    padding: 20px;
    scroll-behavior: smooth;
}

.chat-function-selector {
    padding: 15px;
    display: flex;
    justify-content: center;
    border-top: 1px solid #f0f0f0;
    background-color: #fff;
}
</style>