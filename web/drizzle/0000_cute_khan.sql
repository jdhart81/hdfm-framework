CREATE TABLE `projects` (
	`id` text PRIMARY KEY NOT NULL,
	`owner` text NOT NULL,
	`name` text NOT NULL,
	`data` text NOT NULL,
	`sources` text NOT NULL,
	`revision` integer NOT NULL,
	`created` text NOT NULL,
	`updated` text NOT NULL
);
--> statement-breakpoint
CREATE INDEX `projects_owner_updated` ON `projects` (`owner`,`updated`);--> statement-breakpoint
CREATE TABLE `scenarios` (
	`id` text PRIMARY KEY NOT NULL,
	`project` text NOT NULL,
	`owner` text NOT NULL,
	`name` text NOT NULL,
	`revision` integer NOT NULL,
	`result` text NOT NULL,
	`created` text NOT NULL
);
--> statement-breakpoint
CREATE INDEX `scenarios_project_owner` ON `scenarios` (`project`,`owner`);