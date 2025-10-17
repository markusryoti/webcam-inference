<script setup lang="ts">
import type { NavigationMenuItem } from "@nuxt/ui";

const items: NavigationMenuItem[][] = [
  [
    {
      label: "Connect",
      icon: "i-lucide-house",
      href: "/",
    },
    {
      label: "Results",
      icon: "i-lucide-inbox",
      badge: "4",
      href: "/results",
    },
    {
      label: "Configuration",
      icon: "i-lucide-settings",
      defaultOpen: true,
      children: [
        {
          label: "General",
        },
      ],
    },
  ],
];
</script>

<template>
  <UDashboardSidebar
    collapsible
    resizable
    :ui="{ footer: 'border-t border-default' }"
  >
    <template #header="{ collapsed }">
      <span v-if="!collapsed" class="text-white text-sm font-bold"
        >Detection
      </span>
      <UIcon
        v-else
        name="i-simple-icons-nuxtdotjs"
        class="size-5 text-primary mx-auto"
      />
      <UDashboardSidebarCollapse />
    </template>

    <template #default="{ collapsed }">
      <UNavigationMenu
        :collapsed="collapsed"
        :items="items[0]"
        orientation="vertical"
      />

      <UNavigationMenu
        :collapsed="collapsed"
        :items="items[1]"
        orientation="vertical"
        class="mt-auto"
      />
    </template>

    <template #footer="{ collapsed }">
      <UButton
        :avatar="{
          src: 'https://github.com/benjamincanac.png',
        }"
        :label="collapsed ? undefined : 'Benjamin'"
        color="neutral"
        variant="ghost"
        class="w-full"
        :block="collapsed"
      />
    </template>
  </UDashboardSidebar>
</template>
