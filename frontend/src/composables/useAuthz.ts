import { computed, shallowRef, watch } from 'vue'
import { useLogto } from '@logto/vue'
import { API_RESOURCE } from '@/auth/logto'

export type AuthzRole = 'user' | 'admin' | 'root'

interface TokenClaims {
  sub?: string
  organization_id?: string
  org_id?: string
  organizations?: string[]
  organization_roles?: string[]
  roles?: string[] | string
  role?: string[] | string
  [key: string]: unknown
}

interface AuthzState {
  userId: string
  orgId: string | null
  organizations: string[]
  organizationRoles: string[]
  role: AuthzRole
  loaded: boolean
}

const roleNames: Record<AuthzRole, string> = {
  user: import.meta.env.VITE_LOGTO_ROLE_USER || 'sci-loop-user',
  admin: import.meta.env.VITE_LOGTO_ROLE_ADMIN || 'sci-loop-admin',
  root: import.meta.env.VITE_LOGTO_ROLE_ROOT || 'sci-loop-root',
}

const priority: Record<AuthzRole, number> = { user: 0, admin: 1, root: 2 }

const state = shallowRef<AuthzState>({
  userId: '',
  orgId: null,
  organizations: [],
  organizationRoles: [],
  role: 'user',
  loaded: false,
})

function asStringList(value: unknown): string[] {
  if (!value) return []
  if (typeof value === 'string') return value.split(/[,\s]+/).filter(Boolean)
  if (Array.isArray(value)) return value.map(String).filter(Boolean)
  return [String(value)]
}

function parseJwtPayload(token: string): TokenClaims {
  const payload = token.split('.')[1]
  if (!payload) return {}
  const padded = payload.replace(/-/g, '+').replace(/_/g, '/').padEnd(Math.ceil(payload.length / 4) * 4, '=')
  const json = decodeURIComponent(
    Array.from(atob(padded), (char) => `%${char.charCodeAt(0).toString(16).padStart(2, '0')}`).join(''),
  )
  return JSON.parse(json) as TokenClaims
}

function splitOrgRole(raw: string): [string | null, string] {
  const idx = raw.indexOf(':')
  if (idx < 0) return [null, raw]
  return [raw.slice(0, idx) || null, raw.slice(idx + 1)]
}

function roleFromName(name: string): AuthzRole | null {
  const normalized = name.trim()
  for (const role of ['root', 'admin', 'user'] as const) {
    if (normalized === role || normalized === roleNames[role]) return role
  }
  return null
}

function parseClaims(claims: TokenClaims): AuthzState {
  const organizations = asStringList(claims.organizations)
  const organizationRoles = asStringList(claims.organization_roles)
  let role: AuthzRole = 'user'
  let orgId = (claims.organization_id || claims.org_id || null) as string | null

  for (const raw of [...organizationRoles, ...asStringList(claims.roles), ...asStringList(claims.role)]) {
    const [roleOrg, roleName] = splitOrgRole(raw)
    const parsed = roleFromName(roleName)
    if (parsed && priority[parsed] > priority[role]) {
      role = parsed
      orgId = roleOrg || orgId
    }
  }

  if (!orgId && organizations.length === 1) orgId = organizations[0]

  return {
    userId: claims.sub || '',
    orgId,
    organizations,
    organizationRoles,
    role,
    loaded: true,
  }
}

export function useAuthz() {
  const { getAccessToken, isAuthenticated } = useLogto()

  async function refresh() {
    if (!isAuthenticated.value) {
      state.value = { userId: '', orgId: null, organizations: [], organizationRoles: [], role: 'user', loaded: true }
      return
    }
    try {
      const token = await getAccessToken(API_RESOURCE)
      if (!token) throw new Error('Missing access token')
      state.value = parseClaims(parseJwtPayload(token))
    } catch {
      state.value = { ...state.value, loaded: true }
    }
  }

  watch(isAuthenticated, refresh, { immediate: true })

  return {
    state,
    refresh,
    role: computed(() => state.value.role),
    orgId: computed(() => state.value.orgId),
    isAdmin: computed(() => state.value.role === 'admin' || state.value.role === 'root'),
    isRoot: computed(() => state.value.role === 'root'),
    canUseOrgVisibility: computed(() => Boolean(state.value.orgId) || state.value.role === 'root'),
  }
}
