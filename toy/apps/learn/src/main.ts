import Alpine from "alpinejs";

import "./style.css";
import { COMPONENTS } from "./components/index.ts";

for (const [name, factory] of Object.entries(COMPONENTS)) {
  Alpine.data(name, factory);
}

Alpine.start();
