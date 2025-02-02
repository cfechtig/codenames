import clsx from "clsx";
import { twMerge } from "tailwind-merge";
export const c = (...classes) => twMerge(clsx(...classes));
