// Requires a Cloud or other Grafana instance with Grafana Incident available,
// with a Prometheus datasource provisioned.
//go:build cloud
// +build cloud

// This file contains cloud integration tests that run against a dedicated test instance
// at mcptests.grafana-dev.net. This instance is configured with a minimal setup on the Incident side
// with two incidents created, one minor and one major, and both of them resolved.
// These tests expect this configuration to exist and will skip if the required
// environment variables (GRAFANA_URL, GRAFANA_SERVICE_ACCOUNT_TOKEN or GRAFANA_API_KEY) are not set.
// The GRAFANA_API_KEY variable is deprecated.

package tools

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestAdminToolsIntegration(t *testing.T) {
	t.Run("roles workflow", func(t *testing.T) {
		ctx := createCloudTestContext(t, "Admin", "GRAFANA_URL", "GRAFANA_API_KEY")

		roles, err := listAllRoles(ctx, ListAllRolesParams{})
		require.NoError(t, err)
		assert.NotEmpty(t, roles, "Should return at least one role")

		firstRole := roles[0]
		details, err := getRoleDetails(ctx, GetRoleDetailsParams{RoleUID: *firstRole.UID})
		require.NoError(t, err)
		assert.NotNil(t, details)
		assert.Equal(t, firstRole.UID, details.UID)

		assignments, err := getRoleAssignments(ctx, GetRoleAssignmentsParams{RoleUID: *firstRole.UID})
		require.NoError(t, err)
		assert.NotNil(t, assignments)
	})

	t.Run("users workflow", func(t *testing.T) {
		ctx := createCloudTestContext(t, "Admin", "GRAFANA_URL", "GRAFANA_API_KEY")

		users, err := listUsersByOrg(ctx, ListUsersByOrgParams{})
		require.NoError(t, err)
		assert.NotEmpty(t, users, "Should return at least one user")

		firstUser := users[0]
		userRoles, err := listUserRoles(ctx, ListUserRolesParams{UserIDs: []int64{firstUser.UserID}})
		require.NoError(t, err)
		assert.NotNil(t, userRoles)
	})

	t.Run("teams workflow", func(t *testing.T) {
		ctx := createCloudTestContext(t, "Admin", "GRAFANA_URL", "GRAFANA_API_KEY")

		teams, err := listTeams(ctx, ListTeamsParams{})
		require.NoError(t, err)
		assert.NotNil(t, teams, "Teams result should not be nil")

		if len(teams.Teams) > 0 {
			firstTeam := teams.Teams[0]
			teamRoles, err := listTeamRoles(ctx, ListTeamRolesParams{TeamIDs: []int64{*firstTeam.ID}})
			require.NoError(t, err)
			assert.NotNil(t, teamRoles)
		}
	})

	t.Run("resource description", func(t *testing.T) {
		ctx := createCloudTestContext(t, "Admin", "GRAFANA_URL", "GRAFANA_API_KEY")
		desc, err := getResourceDescription(ctx, GetResourceDescriptionParams{
			ResourceType: "dashboards",
		})
		require.NoError(t, err)
		assert.NotNil(t, desc, "Description should not be nil")
		assert.NotEmpty(t, desc.Assignments, "Should have assignments capabilities")
		assert.NotEmpty(t, desc.Permissions, "Should have permissions capabilities")
	})
}
