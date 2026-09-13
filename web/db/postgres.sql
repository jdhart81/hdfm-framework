-- Run once before starting the PostgreSQL-backed service.
BEGIN;
CREATE TABLE projects (id TEXT PRIMARY KEY, owner TEXT NOT NULL, name TEXT NOT NULL, data TEXT NOT NULL, sources TEXT NOT NULL, revision INTEGER NOT NULL, created TEXT NOT NULL, updated TEXT NOT NULL);
CREATE INDEX projects_owner_updated ON projects(owner,updated);
CREATE TABLE scenarios (id TEXT PRIMARY KEY, project TEXT NOT NULL, owner TEXT NOT NULL, name TEXT NOT NULL, revision INTEGER NOT NULL, result TEXT NOT NULL, created TEXT NOT NULL);
CREATE INDEX scenarios_project_owner ON scenarios(project,owner);
COMMIT;
