resource "azurerm_resource_group" "rg" {
  name     = var.rg
  location = var.location
}

resource "azurerm_kubernetes_cluster" "aks" {
  name                = "ajanach-aks"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  dns_prefix          = "aks-system-node"
  kubernetes_version  = "1.27"

  default_node_pool {
    name       = "ds2v2"
    vm_size    = "Standard_DS2_v2"
    node_count = 1
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin = "kubenet"
    network_policy = "calico"
  }
}