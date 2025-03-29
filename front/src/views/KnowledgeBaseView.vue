<script setup lang="ts">
import { ref, reactive, computed } from 'vue'
import { ElMessage, ElMessageBox } from 'element-plus'
import { Plus, Delete, Edit, Document, FolderAdd, DocumentAdd } from '@element-plus/icons-vue'

interface TextFile {
  id: string;
  name: string;
  content: string;
  createdAt: Date;
  updatedAt?: Date;
  size?: number; // 模拟文件大小，单位KB
}

interface Collection {
  id: string;
  name: string;
  description: string;
  files: TextFile[];
  createdAt: Date;
  updatedAt?: Date;
}

// 示例数据
const collections = ref<Collection[]>([
  {
    id: '1',
    name: '前端开发',
    description: '包含前端开发相关的知识和文档',
    files: [
      { 
        id: '101', 
        name: 'Vue基础教程.txt', 
        content: 'Vue.js是一套用于构建用户界面的渐进式框架。与其它大型框架不同的是，Vue被设计为可以自底向上逐层应用。Vue的核心库只关注视图层，不仅易于上手，还便于与第三方库或既有项目整合。', 
        createdAt: new Date(),
        size: 45
      },
      { 
        id: '102', 
        name: 'React入门指南.txt', 
        content: 'React是一个用于构建用户界面的JavaScript库。React主要用于构建UI，很多人认为React是MVC中的V（视图）。React起源于Facebook的内部项目，用来架设Instagram的网站，并于2013年5月开源。', 
        createdAt: new Date(),
        size: 67
      }
    ],
    createdAt: new Date()
  },
  {
    id: '2',
    name: '机器学习',
    description: '机器学习算法和实践案例',
    files: [
      { 
        id: '201', 
        name: '神经网络入门.txt', 
        content: '神经网络是一种模拟人脑的神经网络的结构和功能的数学模型或计算模型，用于对函数进行估计或近似。神经网络由大量的人工神经元联结进行计算。', 
        createdAt: new Date(),
        size: 89
      }
    ],
    createdAt: new Date()
  }
]);

// 当前选中的集合
const activeCollection = ref<Collection | null>(null);

// 当前选中的文件
const activeFile = ref<TextFile | null>(null);

// 添加集合对话框
const collectionDialogVisible = ref(false);
const newCollection = reactive({
  name: '',
  description: ''
});

// 添加文件对话框
const fileDialogVisible = ref(false);
const newFile = reactive({
  name: '',
  content: ''
});

// 文件预览对话框
const filePreviewVisible = ref(false);

// 选择集合
const selectCollection = (collection: Collection) => {
  activeCollection.value = collection;
  activeFile.value = null; // 清除当前选中的文件
};

// 查看文件内容
const viewFile = (file: TextFile) => {
  activeFile.value = file;
  filePreviewVisible.value = true;
};

// 添加新集合
const addCollection = () => {
  if (!newCollection.name.trim()) {
    ElMessage.warning('集合名称不能为空');
    return;
  }

  const collection: Collection = {
    id: Date.now().toString(),
    name: newCollection.name,
    description: newCollection.description,
    files: [],
    createdAt: new Date()
  };

  collections.value.push(collection);
  collectionDialogVisible.value = false;
  ElMessage.success('集合添加成功');
  
  // 重置表单
  newCollection.name = '';
  newCollection.description = '';
};

// 添加新文件到当前集合
const addFile = () => {
  if (!activeCollection.value) {
    ElMessage.warning('请先选择一个集合');
    return;
  }

  if (!newFile.name.trim()) {
    ElMessage.warning('文件名称不能为空');
    return;
  }

  // 检查文件名是否已存在
  const isExist = activeCollection.value.files.some(file => file.name === newFile.name);
  if (isExist) {
    ElMessage.warning('文件名已存在');
    return;
  }

  const file: TextFile = {
    id: Date.now().toString(),
    name: newFile.name,
    content: newFile.content,
    createdAt: new Date(),
    size: Math.floor(Math.random() * 100) + 10 // 生成10-110KB的随机大小
  };

  activeCollection.value.files.push(file);
  fileDialogVisible.value = false;
  ElMessage.success('文件添加成功');
  
  // 重置表单
  newFile.name = '';
  newFile.content = '';
};

// 删除集合
const deleteCollection = (collection: Collection) => {
  ElMessageBox.confirm(`确定要删除集合 "${collection.name}" 吗？`, '警告', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    type: 'warning',
  })
    .then(() => {
      const index = collections.value.findIndex(c => c.id === collection.id);
      if (index !== -1) {
        collections.value.splice(index, 1);
        if (activeCollection.value?.id === collection.id) {
          activeCollection.value = null;
        }
        ElMessage.success('集合已删除');
      }
    })
    .catch(() => {
      // 用户取消删除
    });
};

// 删除文件
const deleteFile = (file: TextFile) => {
  if (!activeCollection.value) return;

  ElMessageBox.confirm(`确定要删除文件 "${file.name}" 吗？`, '警告', {
    confirmButtonText: '确定',
    cancelButtonText: '取消',
    type: 'warning',
  })
    .then(() => {
      const index = activeCollection.value!.files.findIndex(f => f.id === file.id);
      if (index !== -1) {
        activeCollection.value!.files.splice(index, 1);
        if (activeFile.value?.id === file.id) {
          activeFile.value = null;
        }
        ElMessage.success('文件已删除');
      }
    })
    .catch(() => {
      // 用户取消删除
    });
};

// 统计信息
const statistics = computed(() => {
  return {
    collectionsCount: collections.value.length,
    filesCount: collections.value.reduce((acc, collection) => acc + collection.files.length, 0),
    totalSize: collections.value.reduce(
      (acc, collection) => acc + collection.files.reduce((sum, file) => sum + (file.size || 0), 0), 
      0
    )
  };
});
</script>

<template>
  <div class="knowledge-base-container">
    <div class="knowledge-base-header">
      <h1>知识库管理</h1>
      <div class="header-actions">
        <el-button type="primary" @click="collectionDialogVisible = true" :icon="Plus">
          添加知识集合
        </el-button>
      </div>
    </div>

    <div class="knowledge-base-content">
      <!-- 统计卡片 -->
      <div class="statistics-section">
        <el-row :gutter="20">
          <el-col :span="8">
            <el-card shadow="hover" class="statistic-card">
              <template #header>
                <div class="statistic-header">
                  <span>知识集合</span>
                </div>
              </template>
              <div class="statistic-value">{{ statistics.collectionsCount }}</div>
            </el-card>
          </el-col>
          <el-col :span="8">
            <el-card shadow="hover" class="statistic-card">
              <template #header>
                <div class="statistic-header">
                  <span>文本文件</span>
                </div>
              </template>
              <div class="statistic-value">{{ statistics.filesCount }}</div>
            </el-card>
          </el-col>
          <el-col :span="8">
            <el-card shadow="hover" class="statistic-card">
              <template #header>
                <div class="statistic-header">
                  <span>总容量</span>
                </div>
              </template>
              <div class="statistic-value">{{ statistics.totalSize }} KB</div>
            </el-card>
          </el-col>
        </el-row>
      </div>

      <div class="main-content">
        <el-row :gutter="20">
          <!-- 左侧集合列表 -->
          <el-col :span="8">
            <el-card class="collections-list">
              <template #header>
                <div class="card-header">
                  <span>知识集合列表</span>
                </div>
              </template>

              <div v-if="collections.length === 0" class="empty-state">
                <el-empty description="暂无知识集合">
                  <el-button type="primary" @click="collectionDialogVisible = true">添加集合</el-button>
                </el-empty>
              </div>

              <el-menu v-else mode="vertical" class="collections-menu">
                <el-menu-item 
                  v-for="collection in collections" 
                  :key="collection.id"
                  :index="collection.id"
                  @click="selectCollection(collection)"
                  :class="{ active: activeCollection?.id === collection.id }"
                >
                  <div class="collection-item">
                    <div class="collection-info">
                      <el-icon><FolderAdd /></el-icon>
                      <span class="collection-name">{{ collection.name }}</span>
                      <span class="collection-files-count">{{ collection.files.length }} 文件</span>
                    </div>
                    <div class="collection-actions">
                      <el-button 
                        type="danger" 
                        size="small" 
                        :icon="Delete"
                        @click.stop="deleteCollection(collection)"
                        circle
                      />
                    </div>
                  </div>
                </el-menu-item>
              </el-menu>
            </el-card>
          </el-col>

          <!-- 右侧文件列表 -->
          <el-col :span="16">
            <el-card class="files-list">
              <template #header>
                <div class="card-header">
                  <span>
                    {{ activeCollection ? `${activeCollection.name} 的文件` : '选择一个集合查看文件' }}
                  </span>
                  <el-button 
                    v-if="activeCollection" 
                    type="primary" 
                    size="small" 
                    @click="fileDialogVisible = true"
                    :icon="Plus"
                  >
                    添加文件
                  </el-button>
                </div>
              </template>

              <div v-if="!activeCollection" class="empty-state">
                <el-empty description="请选择一个知识集合查看文件"></el-empty>
              </div>

              <div v-else-if="activeCollection.files.length === 0" class="empty-state">
                <el-empty description="该集合中暂无文件">
                  <el-button type="primary" @click="fileDialogVisible = true">添加文件</el-button>
                </el-empty>
              </div>

              <el-table v-else :data="activeCollection.files" style="width: 100%">
                <el-table-column label="文件名" prop="name" min-width="200">
                  <template #default="scope">
                    <div class="file-name">
                      <el-icon><Document /></el-icon>
                      {{ scope.row.name }}
                    </div>
                  </template>
                </el-table-column>
                <el-table-column label="大小" prop="size" width="120">
                  <template #default="scope">
                    {{ scope.row.size }} KB
                  </template>
                </el-table-column>
                <el-table-column label="创建时间" prop="createdAt" width="180">
                  <template #default="scope">
                    {{ new Date(scope.row.createdAt).toLocaleString() }}
                  </template>
                </el-table-column>
                <el-table-column label="操作" width="150">
                  <template #default="scope">
                    <el-button @click="viewFile(scope.row)" type="primary" size="small" text>
                      查看
                    </el-button>
                    <el-button @click="deleteFile(scope.row)" type="danger" size="small" text>
                      删除
                    </el-button>
                  </template>
                </el-table-column>
              </el-table>
            </el-card>
          </el-col>
        </el-row>
      </div>
    </div>

    <!-- 添加集合对话框 -->
    <el-dialog
      v-model="collectionDialogVisible"
      title="添加知识集合"
      width="500px"
    >
      <el-form :model="newCollection" label-position="top">
        <el-form-item label="集合名称">
          <el-input v-model="newCollection.name" placeholder="请输入集合名称" />
        </el-form-item>
        <el-form-item label="集合描述">
          <el-input 
            v-model="newCollection.description" 
            type="textarea" 
            rows="4"
            placeholder="请输入集合描述"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="collectionDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="addCollection">确认添加</el-button>
      </template>
    </el-dialog>

    <!-- 添加文件对话框 -->
    <el-dialog
      v-model="fileDialogVisible"
      title="添加文本文件"
      width="600px"
    >
      <el-form :model="newFile" label-position="top">
        <el-form-item label="文件名称">
          <el-input v-model="newFile.name" placeholder="请输入文件名称，例如：文件名.txt" />
        </el-form-item>
        <el-form-item label="文件内容">
          <el-input 
            v-model="newFile.content" 
            type="textarea" 
            rows="10"
            placeholder="请输入文件内容"
          />
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="fileDialogVisible = false">取消</el-button>
        <el-button type="primary" @click="addFile">确认添加</el-button>
      </template>
    </el-dialog>

    <!-- 文件预览对话框 -->
    <el-dialog
      v-model="filePreviewVisible"
      :title="activeFile?.name || '文件预览'"
      width="60%"
      destroy-on-close
    >
      <div class="file-preview">
        <pre>{{ activeFile?.content }}</pre>
      </div>
    </el-dialog>
  </div>
</template>

<style scoped>
.knowledge-base-container {
  padding: 30px;
  max-width: 1400px;
  margin: 0 auto;
}

.knowledge-base-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 30px;
}

.knowledge-base-header h1 {
  margin: 0;
  color: #303133;
  font-size: 24px;
}

.statistics-section {
  margin-bottom: 30px;
}

.statistic-card {
  text-align: center;
  height: 100%;
}

.statistic-header {
  font-size: 16px;
  color: #606266;
}

.statistic-value {
  font-size: 32px;
  font-weight: bold;
  color: #409eff;
}

.main-content {
  margin-top: 20px;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.collections-list, .files-list {
  height: 500px;
}

.empty-state {
  height: 450px;
  display: flex;
  justify-content: center;
  align-items: center;
}

.collections-menu {
  border-right: none;
}

.collection-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.collection-info {
  display: flex;
  align-items: center;
  gap: 10px;
}

.collection-name {
  font-weight: 500;
}

.collection-files-count {
  font-size: 12px;
  color: #909399;
  margin-left: 10px;
}

.collection-actions {
  opacity: 0;
  transition: opacity 0.2s;
}

.el-menu-item:hover .collection-actions {
  opacity: 1;
}

.file-name {
  display: flex;
  align-items: center;
  gap: 8px;
}

.file-preview {
  background: #f5f7fa;
  border: 1px solid #ebeef5;
  border-radius: 4px;
  padding: 20px;
  max-height: 400px;
  overflow-y: auto;
}

.file-preview pre {
  margin: 0;
  white-space: pre-wrap;
  word-break: break-word;
}
</style>
