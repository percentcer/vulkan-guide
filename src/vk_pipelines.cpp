#include <vk_pipelines.h>
#include <fstream>
#include <vk_initializers.h>

bool vkutil::load_shader_module(const char* filePath, VkDevice device, VkShaderModule* outShaderModule)
{
	// `ate` opens file "at end", we use this to determine file size
	std::ifstream file(filePath, std::ios::ate | std::ios::binary);
	if (!file.is_open()) { return false; }

	size_t fileSize = (size_t)file.tellg();

	std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));

	// rewind so we can actually read the data
	file.seekg(0);
	file.read((char*)buffer.data(), fileSize);
	file.close();

	VkShaderModuleCreateInfo createInfo{.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .pNext = nullptr };
	createInfo.codeSize = buffer.size() * sizeof(uint32_t);
	createInfo.pCode = buffer.data();

	VkShaderModule module;
	if (vkCreateShaderModule(device, &createInfo, nullptr, &module) != VK_SUCCESS) {
		return false;
	}
	*outShaderModule = module;
	return true;
}
