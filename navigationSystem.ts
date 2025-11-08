import { Entity } from "@serenityjs/core";
import { BlockPosition, Vector3f, Rotation } from "@serenityjs/protocol";

/**
 * Node used in A* algorithm
 */
interface PathNode {
  position: BlockPosition;
  g: number; // Cost from start
  h: number; // Heuristic (estimated distance to goal)
  f: number; // f = g + h
  parent: PathNode | null;
}

/**
 * Navigation state for an entity
 */
interface NavigationState {
  path: Vector3f[];
  currentIndex: number;
  intervalId: NodeJS.Timeout;
}

/**
 * Map of active navigations
 */
const activeNavigations = new Map<Entity, NavigationState>();

/**
 * A* pathfinding navigation system
 */
export class NavigationSystem {
  /**
   * Navigates entity to destination automatically
   */
  public static navigateTo(
    entity: Entity,
    destination: Vector3f,
    speed: number = 0.2
  ): boolean {
    this.stopNavigation(entity);

    const path = this.findPath(entity, destination);
    if (!path || path.length === 0) {
      return false;
    }

    const state: NavigationState = {
      path,
      currentIndex: 0,
      intervalId: null as any
    };

    const intervalId = setInterval(() => {
      if (!entity.isAlive || !entity.dimension) {
        this.stopNavigation(entity);
        return;
      }

      if (state.currentIndex >= state.path.length) {
        this.stopNavigation(entity);
        return;
      }

      const target = state.path[state.currentIndex];
      if (!target) {
        this.stopNavigation(entity);
        return;
      }

      const currentPos = entity.position;
      const dx = target.x - currentPos.x;
      const dy = target.y - currentPos.y;
      const dz = target.z - currentPos.z;
      const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);

      if (distance < 0.5) {
        state.currentIndex++;
        if (state.currentIndex >= state.path.length) {
          this.stopNavigation(entity);
          return;
        }
      } else {
        const normalizedDistance = Math.max(distance, 0.01);
        entity.velocity.x = (dx / normalizedDistance) * speed;
        entity.velocity.y = (dy / normalizedDistance) * speed;
        entity.velocity.z = (dz / normalizedDistance) * speed;
        this.updateEntityRotation(entity, target);
      }
    }, 50);

    state.intervalId = intervalId;
    activeNavigations.set(entity, state);
    return true;
  }

  /**
   * Stops navigation for an entity
   */
  public static stopNavigation(entity: Entity): void {
    const state = activeNavigations.get(entity);
    if (state) {
      clearInterval(state.intervalId);
      activeNavigations.delete(entity);
      entity.velocity.x = 0;
      entity.velocity.y = 0;
      entity.velocity.z = 0;
    }
  }

  /**
   * Checks if entity is navigating
   */
  public static isNavigating(entity: Entity): boolean {
    return activeNavigations.has(entity);
  }

  /**
   * Finds path from start to destination using A*
   */
  public static findPath(
    entity: Entity,
    destination: Vector3f
  ): Vector3f[] | null {
    const dimension = entity.dimension;
    const startPos = BlockPosition.fromVector3f(entity.position);
    const endPos = BlockPosition.fromVector3f(destination);

    if (this.positionsEqual(startPos, endPos)) {
      return [destination];
    }

    const openSet: PathNode[] = [];
    const closedSet = new Set<string>();

    const startNode: PathNode = {
      position: startPos,
      g: 0,
      h: this.heuristic(startPos, endPos),
      f: 0,
      parent: null
    };
    startNode.f = startNode.g + startNode.h;
    openSet.push(startNode);

    const maxIterations = 10000;
    let iterations = 0;

    while (openSet.length > 0 && iterations < maxIterations) {
      iterations++;

      // Find node with lowest f (optimized: track min index)
      let currentIndex = 0;
      let minF = openSet[0]?.f ?? Infinity;
      for (let i = 1; i < openSet.length; i++) {
        const nodeF = openSet[i]?.f ?? Infinity;
        if (nodeF < minF) {
          minF = nodeF;
          currentIndex = i;
        }
      }

      const current = openSet.splice(currentIndex, 1)[0];
      if (!current) break;

      const currentKey = this.positionKey(current.position);

      if (this.positionsEqual(current.position, endPos)) {
        return this.reconstructPath(current);
      }

      closedSet.add(currentKey);

      const neighbors = this.getNeighbors(current.position);
      for (const neighborPos of neighbors) {
        const neighborKey = this.positionKey(neighborPos);

        if (closedSet.has(neighborKey)) {
          continue;
        }

        const moveCost = this.getMoveCost(
          current.position,
          neighborPos,
          dimension
        );

        if (moveCost === Infinity) {
          continue;
        }

        const tentativeG = current.g + moveCost;
        let neighborNode = openSet.find(
          (n) => this.positionsEqual(n.position, neighborPos)
        );

        if (!neighborNode) {
          neighborNode = {
            position: neighborPos,
            g: tentativeG,
            h: this.heuristic(neighborPos, endPos),
            f: 0,
            parent: current
          };
          neighborNode.f = neighborNode.g + neighborNode.h;
          openSet.push(neighborNode);
        } else if (tentativeG < neighborNode.g) {
          neighborNode.g = tentativeG;
          neighborNode.f = neighborNode.g + neighborNode.h;
          neighborNode.parent = current;
        }
      }
    }

    return null;
  }

  /**
   * Reconstructs path from final node
   */
  private static reconstructPath(node: PathNode): Vector3f[] {
    const path: Vector3f[] = [];
    let current: PathNode | null = node;

    while (current !== null) {
      path.unshift(
        new Vector3f(
          current.position.x + 0.5,
          current.position.y + 0.5,
          current.position.z + 0.5
        )
      );
      current = current.parent;
    }

    return path;
  }

  /**
   * Gets valid neighbors of a position
   */
  private static getNeighbors(position: BlockPosition): BlockPosition[] {
    const neighbors: BlockPosition[] = [];
    const directions = [
      { x: 1, y: 0, z: 0 },
      { x: -1, y: 0, z: 0 },
      { x: 0, y: 0, z: 1 },
      { x: 0, y: 0, z: -1 },
      { x: 1, y: 0, z: 1 },
      { x: 1, y: 0, z: -1 },
      { x: -1, y: 0, z: 1 },
      { x: -1, y: 0, z: -1 },
      { x: 0, y: 1, z: 0 },
      { x: 0, y: -1, z: 0 }
    ];

    for (const dir of directions) {
      neighbors.push(
        new BlockPosition(
          position.x + dir.x,
          position.y + dir.y,
          position.z + dir.z
        )
      );
    }

    return neighbors;
  }

  /**
   * Calculates movement cost between two positions
   * Returns Infinity if movement is invalid
   */
  private static getMoveCost(
    from: BlockPosition,
    to: BlockPosition,
    dimension: any
  ): number {
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const dz = to.z - from.z;

    // Vertical movement up (jump)
    if (dy === 1) {
      const blockAtDest = dimension.getBlock(to);
      const blockAboveDest = dimension.getBlock(
        new BlockPosition(to.x, to.y + 1, to.z)
      );
      const blockBelowDest = dimension.getBlock(
        new BlockPosition(to.x, to.y - 1, to.z)
      );

      if (!blockAtDest.isSolid && !blockAboveDest.isSolid && blockBelowDest.isSolid) {
        return 1.5;
      }
      return Infinity;
    }

    // Vertical movement down
    if (dy === -1) {
      const blockAtDest = dimension.getBlock(to);
      const blockAboveDest = dimension.getBlock(
        new BlockPosition(to.x, to.y + 1, to.z)
      );
      const blockBelowDest = dimension.getBlock(
        new BlockPosition(to.x, to.y - 1, to.z)
      );

      if (!blockAtDest.isSolid && !blockAboveDest.isSolid && blockBelowDest.isSolid) {
        return 1.2;
      }
      return Infinity;
    }

    // Horizontal or diagonal movement
    if (dy === 0) {
      const blockAt = dimension.getBlock(to);
      const blockAbove = dimension.getBlock(
        new BlockPosition(to.x, to.y + 1, to.z)
      );
      const blockBelow = dimension.getBlock(
        new BlockPosition(to.x, to.y - 1, to.z)
      );

      if (blockAt.isSolid || blockAbove.isSolid || !blockBelow.isSolid) {
        return Infinity;
      }

      const isDiagonal = dx !== 0 && dz !== 0;
      return isDiagonal ? 1.414 : 1.0;
    }

    return Infinity;
  }

  /**
   * Heuristic: Manhattan distance
   */
  private static heuristic(
    from: BlockPosition,
    to: BlockPosition
  ): number {
    return (
      Math.abs(from.x - to.x) +
      Math.abs(from.y - to.y) +
      Math.abs(from.z - to.z)
    );
  }

  /**
   * Compares two positions
   */
  private static positionsEqual(
    a: BlockPosition,
    b: BlockPosition
  ): boolean {
    return a.x === b.x && a.y === b.y && a.z === b.z;
  }

  /**
   * Generates unique key for position (for Set/Map)
   */
  private static positionKey(position: BlockPosition): string {
    return `${position.x},${position.y},${position.z}`;
  }

  /**
   * Updates entity rotation to look at target
   */
  private static updateEntityRotation(entity: Entity, target: Vector3f): void {
    const currentPos = entity.position;
    const dx = target.x - currentPos.x;
    const dy = target.y - currentPos.y;
    const dz = target.z - currentPos.z;
    const horizontalDistance = Math.sqrt(dx * dx + dz * dz);

    if (horizontalDistance < 0.01) {
      return;
    }

    let yaw = Math.atan2(dz, dx) * (180 / Math.PI) - 90;
    while (yaw > 180) yaw -= 360;
    while (yaw < -180) yaw += 360;

    let pitch = 0;
    if (Math.abs(dy) > 0.01) {
      pitch = -Math.atan2(dy, horizontalDistance) * (180 / Math.PI);
      if (pitch > 90) pitch = 90;
      if (pitch < -90) pitch = -90;
    }

    entity.setRotation(new Rotation(yaw, pitch, yaw));
  }
}
