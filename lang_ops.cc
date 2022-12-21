#include <mutex>
#include <memory>
#include <onnxruntime_cxx_api.h>
#include "lua_op.h"


struct OrtCustomOpDomainDeleter {
  explicit OrtCustomOpDomainDeleter(const OrtApi* ort_api) {
    ort_api_ = ort_api;
  }
  void operator()(OrtCustomOpDomain* domain) const {
    ort_api_->ReleaseCustomOpDomain(domain);
  }

  const OrtApi* ort_api_;
};

using OrtCustomOpDomainUniquePtr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>;
static std::vector<OrtCustomOpDomainUniquePtr> ort_custom_op_domain_container;
static std::mutex ort_custom_op_domain_mutex;

static void AddOrtCustomOpDomainToContainer(OrtCustomOpDomain* domain, const OrtApi* ort_api) {
  std::lock_guard<std::mutex> lock(ort_custom_op_domain_mutex);
  auto ptr = std::unique_ptr<OrtCustomOpDomain, OrtCustomOpDomainDeleter>(domain, OrtCustomOpDomainDeleter(ort_api));
  ort_custom_op_domain_container.push_back(std::move(ptr));
}

static const char* c_OpDomain = "lang.lua";
static const LuaOp c_LuaOp;

extern "C" OrtStatus* ORT_API_CALL RegisterCustomOps(OrtSessionOptions* options, const OrtApiBase* api) { 
  OrtCustomOpDomain* domain = nullptr; 
  const OrtApi* ortApi = api->GetApi(ORT_API_VERSION); 

  if (auto status = ortApi->CreateCustomOpDomain(c_OpDomain, &domain)) { 
    return status; 
  } 

  AddOrtCustomOpDomainToContainer(domain, ortApi);

  if (auto status = ortApi->CustomOpDomain_Add(domain, &c_LuaOp)) { 
    return status; 
  }

  if (auto status = ortApi->AddCustomOpDomain(options, domain)) {
    return status;
  }

  return nullptr;
}